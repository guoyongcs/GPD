
import os
import time
import sys
import math
import glob
import yaml

from timm.utils import *
from timm.models import resume_checkpoint, load_checkpoint
from torch.utils.data import Dataset, DataLoader, distributed
from torch.utils.data.dataset import Subset
import torch.distributed as dist
import logging
import myutils
from contextlib import suppress
import datetime
from collections import OrderedDict

# sys.path.append("../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import *

import argparse
import h5py
# from vgg_loss import *
# from discriminator_arch import *

# import moxing.pytorch as mox
# from moxing.pytorch.utils.hyper_param_flags import get_flag

from torch.utils.tensorboard import SummaryWriter

from bisect import bisect_left
from torch.utils.data.dataloader import default_collate
from transforms import get_mixup_cutmix
from torchvision.transforms.functional import InterpolationMode
import torchvision
import presets
from models import *
from timm.utils.clip_grad import dispatch_clip_grad
from gpd_utils import gpd_init, forward_tiny, forward_expanded, enhance_with_es, check_clean_grad, correct_grad
from reviewkd_utils import build_review_kd, hcl
import torchvision.transforms as transforms
from orepa_ft import transfer2orepa
from torchvision.models.convnext import convnext_tiny
from torchvision.models.vision_transformer import vit_l_16


try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss



class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        ttt = np.copy(nump_array)
        tensor[i] += torch.from_numpy(ttt)

    return tensor, targets


def flow_warp(img, flow):

    # assert img.size()[-2:] == flow.size()[-2:]
    bs, _, h, w = flow.size()
    u = flow[:,0,:,:] # NCHW
    v = flow[:,1,:,:]
    
    X = torch.clamp(u, min=0., max=w-1.0)
    Y = torch.clamp(v, min=0., max=h-1.0)

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)

    grid_tf = torch.stack((X,Y), dim=3)
    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode='zeros', mode='nearest')

    return img_tf


def adjust_learning_rate(args, optimizer, epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""

    """Warmup"""
#    lr = args.learning_rate
#    if epoch < 5:
#        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr
#        return
    lr = args.learning_rate
    if args.lr_warmup_epochs > 0 and epoch < args.lr_warmup_epochs:
        lr = lr * min(epoch + 1, args.lr_warmup_epochs + 1e-6) / (args.lr_warmup_epochs + 1e-6)
    else:
        for i in args.lr_adjust_step:
            if i <= epoch:
                lr *= 0.1
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr

    return lr


class Trainer():

    def __init__(self, config_dict_, _logger):
        self.config_dict = config_dict_
        self._logger = _logger
        name_model = eval(self.config_dict.model)  # self.GetNameModel(args['model'])
        object_model = name_model()
        if self.config_dict.local_rank == 0:
            self._logger.info('Model %s created, param count: %d' %
                         (name_model.__name__, sum([m.numel() for m in object_model.parameters()])))
            self.writer = SummaryWriter(self.config_dict.tensorboard_dir if self.config_dict.tensorboard_dir else self.config_dict.checkpoint_dir)

        # load pretrained tiny model
        if self.config_dict.tiny_ckpt_load:
            self._logger.info(f'Load pretrained tiny model from {self.config_dict.tiny_ckpt_load}')
            load_checkpoint(object_model, self.config_dict.tiny_ckpt_load, use_ema=self.config_dict.tiny_eval_model_ema, strict=False)

        if self.config_dict.use_orepa:
            object_model = transfer2orepa(object_model, train_from_scratch=self.config_dict.tiny_ckpt_load is None, num_branches_orepa=self.config_dict.num_branches_orepa)

        test_input = torch.rand(1, 3, 224, 224)
        if not self.config_dict.skip_es:
            if self.config_dict.model == 'MobileNet':
                first_module_name_list = ['model.0.0']
            elif 'resnet' in self.config_dict.model:
                first_module_name_list = ['conv1']

            # exclude_module_name_list = ['identity_beginner']
            object_model = gpd_init(object_model, first_module_name_list=first_module_name_list, last_module_name_list=['fc'], test_input=[test_input], gpd_ratio=self.config_dict.gpd_ratio)
            # assert False, sum([m.numel() for m in object_model.parameters()])
        else:
            self.config_dict.gpd_ratio = 1

        if self.config_dict.use_reviewkd:
            object_model = build_review_kd(self.config_dict.model, object_model, expand_ratio=self.config_dict.gpd_ratio if not self.config_dict.skip_es else 1)

        # load pretrainded expanded model but start from epoch 0
        if self.config_dict.ckpt_load:
            load_checkpoint(object_model, self.config_dict.ckpt_load, use_ema=self.config_dict.eval_model_ema)

        object_model.cuda()
        # rebuild optimizer
        self.optimizer = self.build_optimizer(self.config_dict, object_model)
        # object_model, self.optimizer, self.amp_autocast = self.setup_amp(object_model, self.optimizer)

        # auto resume
        self.start_epoch = 0
        checkpoint_path = os.path.join(self.config_dict.checkpoint_dir, 'last.pth.tar')
        if os.path.exists(checkpoint_path):
            # if self.config_dict.resume is not None and os.path.exists(self.config_dict.resume):
            resume_epoch = resume_checkpoint(
                object_model, checkpoint_path,
                optimizer=None if self.config_dict.no_resume_opt else self.optimizer,
                # loss_scaler=None if self.config_dict.no_resume_opt else loss_scaler,
                log_info=self.config_dict.local_rank == 0)
            self.start_epoch = resume_epoch
            self._logger.info(f'Resume training from epoch {resume_epoch}')

        # resume with specific path
        if self.config_dict.resume is not None and os.path.exists(self.config_dict.resume):
            resume_epoch = resume_checkpoint(
                object_model, self.config_dict.resume,
                optimizer=None if self.config_dict.no_resume_opt else self.optimizer,
                # loss_scaler=None if self.config_dict.no_resume_opt else loss_scaler,
                log_info=self.config_dict.local_rank == 0)
            self.start_epoch = resume_epoch
            self._logger.info(f'Resume training from epoch {resume_epoch}')

        self.config_dict.resume = checkpoint_path if self.config_dict.resume is None and os.path.exists(checkpoint_path) else self.config_dict.resume

        # move model to GPU, enable channels last layout if set
        object_model.cuda()
        if self.config_dict.channels_last:
            object_model = object_model.to(memory_format=torch.channels_last)

        # setup exponential moving average of model weights, SWA could be used here too
        self.model_ema = self.build_ema(object_model)

        # setup scheduler
        self.config_dict.lr_scheduler = self.config_dict.lr_scheduler.lower()
        self.config_dict.lr_scheduler = self.config_dict.lr_scheduler
        if self.config_dict.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config_dict.lr_step_size, gamma=self.config_dict.lr_gamma)
        if self.config_dict.lr_scheduler == "multisteplr":
            main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30 - self.config_dict.lr_warmup_epochs, 60 - self.config_dict.lr_warmup_epochs], gamma=self.config_dict.lr_gamma)
        elif self.config_dict.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config_dict.epoch_init - self.config_dict.lr_warmup_epochs, eta_min=self.config_dict.lr_min
            )
        elif self.config_dict.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config_dict.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{self.config_dict.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        # if self.config_dict.lr_warmup_epochs > 0:
        #     if self.config_dict.lr_warmup_method == "linear":
        #         warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #             self.optimizer, start_factor=self.config_dict.lr_warmup_decay, total_iters=self.config_dict.lr_warmup_epochs
        #         )
        #     elif self.config_dict.lr_warmup_method == "constant":
        #         warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
        #             self.optimizer, factor=self.config_dict.lr_warmup_decay, total_iters=self.config_dict.lr_warmup_epochs
        #         )
        #     else:
        #         raise RuntimeError(
        #             f"Invalid warmup lr method '{self.config_dict.lr_warmup_method}'. Only linear and constant are supported."
        #         )
        #     self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        #         self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[self.config_dict.lr_warmup_epochs]
        #     )
        # else:
        #     self.lr_scheduler = main_lr_scheduler

        # build teacher
        if self.config_dict.teacher is not None:
            name_model = eval(self.config_dict.teacher)  # self.GetNameModel(args['model'])
            teacher = name_model()
            for param in teacher.parameters():
                param.requires_grad = False
            checkpoint = torch.load(self.config_dict.teacher_weight, map_location='cpu')
            teacher.load_state_dict(checkpoint, strict=False)
            teacher = teacher.cuda()
            self._logger.info(f'Load teacher model from {self.config_dict.teacher_weight}')
        else:
            teacher = None
        self.teacher = teacher

        # setup distributed training
        self.model = self.setup_ddp(object_model)
        self.saver = self.build_saver(self.model, self.model_ema, self.optimizer)

        self.make_dataset_imagenet()
        self.batch_cnt=len(self.data_loader)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config_dict.label_smoothing).cuda()

    def build_optimizer(self, args, object_model):
        custom_keys_weight_decay = []
        if args.bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
        if args.transformer_embedding_decay is not None:
            for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
                custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
        parameters = myutils.set_weight_decay(
            object_model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )

        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316,
                alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

        return optimizer


    def build_ema(self, model):
        model_ema = None
        if self.config_dict.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEmaV2(
                model, decay=self.config_dict.model_ema_decay, device='cpu' if self.config_dict.model_ema_force_cpu else None)
            if self.config_dict.resume is None and self.config_dict.ckpt_load is not None:
                try:
                    load_checkpoint(model_ema.module, self.config_dict.ckpt_load, use_ema=self.config_dict.eval_model_ema)
                except:
                    self._logger.info('build ema without loading weights')
            elif self.config_dict.resume is not None and not self.config_dict.not_load_ema:
                try:
                    load_checkpoint(model_ema.module, self.config_dict.resume, use_ema=True)
                except:
                    self._logger.info('build ema without loading weights')
            else:
                self._logger.info('build ema without loading weights')
        return model_ema

    def setup_amp(self, model, optimizer):
        amp_autocast = suppress  # do nothing
        if self.config_dict.use_amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            if self.config_dict.local_rank == 0:
                self._logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif self.config_dict.use_amp == 'native':
            amp_autocast = torch.cuda.amp.autocast
            if self.config_dict.local_rank == 0:
                self._logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            if self.config_dict.local_rank == 0:
                self._logger.info('AMP not enabled. Training in float32.')
        return model, optimizer, amp_autocast

    def setup_ddp(self, model):
        if self.config_dict.distributed:
            if has_apex and self.config_dict.use_amp and self.config_dict.use_amp != 'native':
                # Apex DDP preferred unless native amp is activated
                if self.config_dict.local_rank == 0:
                    self._logger.info("Using NVIDIA APEX DistributedDataParallel.")
                model = ApexDDP(model, delay_allreduce=True, )
            else:
                if self.config_dict.local_rank == 0:
                    self._logger.info("Using native Torch DistributedDataParallel.")
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config_dict.local_rank], find_unused_parameters=True)  # can use device str in Torch >= 1.1
            # NOTE: EMA model does not need to be wrapped by DDP
        return model


    def load_data(self, traindir, valdir, args):
        # Data loading code
        val_resize_size, val_crop_size, train_crop_size = (
            args.val_resize_size,
            args.val_crop_size,
            args.train_crop_size,
        )
        interpolation = InterpolationMode(args.interpolation)

        print("Loading training data")
        st = time.time()

        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        print("Took", time.time() - st)

        print("Loading validation data")
        preprocessing = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
            backend=args.backend,
            use_v2=args.use_v2,
        )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )

        print("Creating data samplers")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size,
                                                                            rank=args.rank)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=args.world_size,
                                                                           rank=args.rank, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        return dataset, dataset_test, train_sampler, test_sampler


    def make_dataset_imagenet(self):
        # build dataset
        train_dir = os.path.join(self.config_dict.train_data_dir, "train")
        val_dir = os.path.join(self.config_dict.train_data_dir, "val")

        val_resize_size, val_crop_size, train_crop_size = (
            self.config_dict.val_resize_size,
            self.config_dict.val_crop_size,
            self.config_dict.train_crop_size,
        )

        trans = transforms.Compose([
            transforms.RandomResizedCrop(train_crop_size),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), Too slow
            # normalize,
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            train_dir, trans
        )
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(val_resize_size),
            transforms.CenterCrop(val_crop_size),
        ]))

        train_sampler, test_sampler = None, None
        if self.config_dict.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        self.data_sampler = train_sampler

        self.data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config_dict.batch_size, shuffle=(train_sampler is None),
            num_workers=self.config_dict.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

        self.valid_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config_dict.batch_size, shuffle=False,
            num_workers=self.config_dict.workers, pin_memory=True,
            sampler=test_sampler,
            collate_fn=fast_collate)


        # dataset, dataset_test, train_sampler, test_sampler = self.load_data(train_dir, val_dir, self.config_dict)
        # self.data_sampler = train_sampler
        # num_classes = len(dataset.classes)
        # mixup_cutmix = get_mixup_cutmix(
        #     mixup_alpha=self.config_dict.mixup_alpha, cutmix_alpha=self.config_dict.cutmix_alpha, num_categories=num_classes, use_v2=self.config_dict.use_v2
        # )
        # if mixup_cutmix is not None:
        #     def collate_fn(batch):
        #         return mixup_cutmix(*default_collate(batch))
        # else:
        #     collate_fn = default_collate
        #
        # self.data_loader = torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=self.config_dict.batch_size,
        #     sampler=self.data_sampler,
        #     num_workers=self.config_dict.workers,
        #     pin_memory=True,
        #     collate_fn=collate_fn,
        # )
        # self.valid_data_loader = torch.utils.data.DataLoader(
        #     dataset_test, batch_size=self.config_dict.batch_size, sampler=test_sampler, num_workers=self.config_dict.workers, pin_memory=True
        # )

    def build_saver(self, model, model_ema, optimizer):
        # setup checkpoint saver
        saver = None
        eval_metric = self.config_dict.eval_metric
        if self.config_dict.rank == 0:
            output_base = self.config_dict.checkpoint_dir if self.config_dict.checkpoint_dir else './output'
            output_dir = output_base
            code_dir = get_outdir(output_dir, 'code')
            myutils.mycopy_tree(os.getcwd(), code_dir)
            decreasing = True if eval_metric == 'loss' else False
            saver = myutils.MyCheckpointSaver(
                model=model, optimizer=optimizer, args=self.config_dict, model_ema=model_ema, amp_scaler=None,
                checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing,
                max_history=self.config_dict.checkpoint_hist)
            if self.config_dict.resume:
                try:
                    saver.load_checkpoint_files()
                except:
                    pass
        return saver

    def compute_loss(self, pred, criterion, target, use_reviewkd=False, review_kd_loss_weight=1.0, logits_teacher=None, fteacher=None, reviewkd_active=True, use_dkd=False, dkd_active=True):
        if use_reviewkd and reviewkd_active:
            fstudent, pred = pred
            loss_reviewkd = hcl(fstudent, fteacher) * review_kd_loss_weight
            loss = criterion(pred, target)
            loss = loss + loss_reviewkd
        else:
            if use_reviewkd:
                fstudent, pred = pred
            loss = criterion(pred, target)
            loss_reviewkd = loss * 0
        # dkd
        if logits_teacher is not None and use_dkd and dkd_active:
            alpha = 1.0
            # 0.5 for resnet18, 2 for mobilenet
            beta = 0.5
            if self.config_dict.model == 'MobileNet':
                beta = 2.0
            temperature = 1.0
            loss_dkd = dkd_loss(
                pred,
                logits_teacher,
                target,
                alpha,
                beta,
                temperature,
            )
            loss = loss + loss_dkd
        else:
            loss_dkd = loss * 0

        return loss, loss_reviewkd, loss_dkd


    def adaptive_lr_based_on_loss(self):
        self.model.eval()

        prefetcher = data_prefetcher(self.data_loader)
        input, target = prefetcher.next()

        if self.teacher is not None:
            with torch.no_grad():
                fteacher, logits_teacher = self.teacher(input, is_feat=True)
        else:
            fteacher = None
            logits_teacher = None

        expanded_result = self.model(input)
        expanded_loss, reviewkd_loss, dkd_loss = self.compute_loss(expanded_result, self.criterion, target, use_reviewkd=self.config_dict.use_reviewkd, review_kd_loss_weight=self.config_dict.review_kd_loss_weight, logits_teacher=logits_teacher, fteacher=fteacher, reviewkd_active=True, use_dkd=self.config_dict.use_dkd, dkd_active=True)

        # train with es
        loss_param_list = [self.criterion, target, self.config_dict.use_reviewkd, self.config_dict.review_kd_loss_weight, logits_teacher, fteacher, True, self.config_dict.use_dkd, True]
        total_loss, kd_loss, tiny_result = enhance_with_es(self.model, input, expanded_result, expanded_loss, self.compute_loss, loss_param_list, kd_loss_type=self.config_dict.kd_loss_type, kd_loss_weight=self.config_dict.kd_ratio, warmup_es_epoch=self.config_dict.warmup_es_epoch)


        lr_scale = expanded_loss / (expanded_loss + total_loss)
        lr_scale = lr_scale.item()

        self.config_dict.learning_rate = self.config_dict.learning_rate * lr_scale


    def train_one_epoch_es(self, epoch):
        self.model.train()
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        acc1_m = AverageMeter()
        expanded_acc1_m = AverageMeter()

        prefetcher = data_prefetcher(self.data_loader)
        input, target = prefetcher.next()
        idx = 0

        end = time.time()
        last_idx = len(self.data_loader)
        num_updates = epoch * len(self.data_loader)

        while input is not None:
            idx += 1

        # for idx, (input, target) in enumerate(self.data_loader):
        #     input, target = input.cuda(), target.cuda()
            last_batch = idx == last_idx
            data_time_m.update(time.time() - end)

            lr = adjust_learning_rate(self.config_dict, self.optimizer, epoch)
            self.optimizer.zero_grad()

            if self.teacher is not None:
                with torch.no_grad():
                    try:
                        fteacher, logits_teacher = self.teacher(input, is_feat=True)
                    except:
                        logits_teacher = self.teacher(input)
                        fteacher = None
            else:
                fteacher = None
                logits_teacher = None

            # expanded_result = self.model(input)
            expanded_result = forward_expanded(self.model, input)

            expanded_loss, reviewkd_loss, dkd_loss = self.compute_loss(expanded_result, self.criterion, target, use_reviewkd=self.config_dict.use_reviewkd, review_kd_loss_weight=self.config_dict.review_kd_loss_weight, logits_teacher=logits_teacher, fteacher=fteacher, reviewkd_active=not self.config_dict.deactive_kd_static_teacher, use_dkd=self.config_dict.use_dkd, dkd_active=not self.config_dict.deactive_kd_static_teacher)


            # expanded_loss, reviewkd_loss, dkd_loss = self.compute_loss(expanded_result, self.criterion, target, use_reviewkd=self.config_dict.use_reviewkd, review_kd_loss_weight=self.config_dict.review_kd_loss_weight, logits_teacher=logits_teacher, fteacher=fteacher, reviewkd_active=self.config_dict.skip_es, use_dkd=self.config_dict.use_dkd, dkd_active=self.config_dict.skip_es)


            # train with es
            if not self.config_dict.skip_es:
                # if self.teacher is not None:
                #     expanded_loss = expanded_loss / 2
                loss_param_list = [self.criterion, target, self.config_dict.use_reviewkd, self.config_dict.review_kd_loss_weight, logits_teacher, fteacher, True, self.config_dict.use_dkd, True]
                total_loss, kd_loss, tiny_result = enhance_with_es(self.model, input, expanded_result, expanded_loss, self.compute_loss, loss_param_list, kd_loss_type=self.config_dict.kd_loss_type, kd_loss_weight=self.config_dict.kd_ratio, warmup_es_epoch=self.config_dict.warmup_es_epoch, current_epoch=epoch, skip_expanded=self.config_dict.skip_expanded)
                # if self.teacher is not None:
                #     total_loss = total_loss / 2
            else:
                total_loss = expanded_loss
                kd_loss = expanded_loss * 0
                tiny_result = expanded_result

            if self.config_dict.use_reviewkd:
                _, tiny_result = tiny_result
                _, expanded_result = expanded_result

            acc1, acc5 = accuracy(tiny_result, target, topk=(1, 5))
            expanded_acc1, expanded_acc5 = accuracy(expanded_result, target, topk=(1, 5))

            if torch.any(torch.isnan(total_loss)):
                # assert False, f'loss becomes nan!!!'
                checkpoint_path = os.path.join(self.config_dict.checkpoint_dir, 'last.pth.tar')
                _ = resume_checkpoint(
                    self.model.module, checkpoint_path,
                    optimizer=self.optimizer,
                    log_info=self.config_dict.local_rank == 0)
                load_checkpoint(self.model_ema.module, checkpoint_path, use_ema=False)
                # self.model_ema = self.build_ema(self.model.module)
                self._logger.info(f'Restart from epoch {epoch}')

                return epoch

            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters=self.model.module.parameters(), max_norm=self.config_dict.clip_grad_norm, norm_type=2.0)

            # whether_use_clip = self.config_dict.clip_grad is not None
            # if whether_use_clip:
            #     dispatch_clip_grad(self.model.module.parameters(), self.config_dict.clip_grad, mode='norm')

            self.optimizer.step()

            if not self.config_dict.distributed:
                losses_m.update(total_loss.item(), input.size(0))
                acc1_m.update(acc1.item(), input.size(0))
                expanded_acc1_m.update(expanded_acc1.item(), input.size(0))

            if self.model_ema is not None:
                self.model_ema.update(self.model)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if self.config_dict.distributed:
                reduced_loss = reduce_tensor(total_loss.data, self.config_dict.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                reduced_acc1 = reduce_tensor(acc1.data, self.config_dict.world_size)
                acc1_m.update(reduced_acc1.item(), input.size(0))
                reduced_expanded_acc1 = reduce_tensor(expanded_acc1.data, self.config_dict.world_size)
                expanded_acc1_m.update(reduced_expanded_acc1.item(), input.size(0))

            if last_batch or idx % self.config_dict.log_interval == 0:
                if self.config_dict.rank == 0:
                    eta_string = str(datetime.timedelta(seconds=int(batch_time_m.avg * (len(self.data_loader) - idx))))
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)] eta: {}  '
                        'Loss: {loss.val:>9.2f} ({loss.avg:>6.2f}) {kd:>6.2f}  '
                        # 'ReviewKD: {reviewkd_loss:>6.2f}  '
                        'GNorm: {grad_norm:>6.2f}  '
                        'Acc: {acc.val:>9.2f} ({acc.avg:>6.2f})  '
                        'E_Acc: {expanded_acc.val:>9.2f} ({expanded_acc.avg:>6.2f})  '
                        'Time: {batch_time.val:.3f}s,  '
                        'LR: {lr:.1e}  '
                        'Data: {data_time.val:.3f}'.format(
                            epoch,
                            idx, len(self.data_loader),
                            100. * idx / last_idx,
                            eta_string,
                            loss=losses_m,
                            kd=kd_loss.data,
                            # reviewkd_loss=reviewkd_loss.data,
                            grad_norm=grad_norm.data,
                            acc=acc1_m,
                            expanded_acc=expanded_acc1_m,
                            batch_time=batch_time_m,
                            lr=lr,
                            data_time=data_time_m))

            end = time.time()
            input, target = prefetcher.next()
            # end for

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg), ('acc1', acc1_m.avg), ('e_acc1', expanded_acc1_m.avg)])


    @torch.no_grad()
    def validate(self, model, log_suffix=''):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        acc1_m = AverageMeter()

        model.eval()

        prefetcher = data_prefetcher(self.valid_data_loader)
        input, target = prefetcher.next()

        end = time.time()
        last_idx = len(self.valid_data_loader)

        idx = 0
        while input is not None:
            idx += 1
            last_batch = idx == last_idx

            if self.teacher is not None:
                with torch.no_grad():
                    try:
                        fteacher, logits_teacher = self.teacher(input, is_feat=True)
                    except:
                        logits_teacher = self.teacher(input)
                        fteacher = None
            else:
                fteacher = None
                logits_teacher = None

            net_result = model(input)
            # compute loss
            total_loss, reviewkd_loss, dkd_loss = self.compute_loss(net_result, self.criterion, target, use_reviewkd=self.config_dict.use_reviewkd, review_kd_loss_weight=self.config_dict.review_kd_loss_weight, logits_teacher=logits_teacher, fteacher=fteacher, reviewkd_active=True, use_dkd=self.config_dict.use_dkd, dkd_active=True)


            if self.config_dict.use_reviewkd:
                _, net_result = net_result

            # compute acc
            acc1, acc5 = accuracy(net_result, target, topk=(1, 5))

            if not self.config_dict.distributed:
                losses_m.update(total_loss.item(), input.size(0))
                acc1_m.update(acc1.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time_m.update(time.time() - end)
            if self.config_dict.distributed:
                reduced_loss = reduce_tensor(total_loss.data, self.config_dict.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                reduced_acc1 = reduce_tensor(acc1.data, self.config_dict.world_size)
                acc1_m.update(reduced_acc1.item(), input.size(0))

            if last_batch or idx % self.config_dict.log_interval == 0:
                if self.config_dict.rank == 0:
                    log_name = 'Test' + log_suffix
                    self._logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})  '.format(
                            log_name, idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, acc=acc1_m))

            end = time.time()
            input, target = prefetcher.next()
            # end for
        if self.config_dict.rank == 0:
            self._logger.info(f"[Validation] Loss:{losses_m.avg:.2f}, Acc:{acc1_m.avg:.2f}%")
        return OrderedDict([('loss', losses_m.avg), ('acc', acc1_m.avg)])


    @torch.no_grad()
    def validate_es(self, model, log_suffix=''):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        acc1_m = AverageMeter()
        acc5_m = AverageMeter()
        expanded_acc1_m = AverageMeter()
        expanded_acc5_m = AverageMeter()

        model.eval()

        prefetcher = data_prefetcher(self.valid_data_loader)
        input, target = prefetcher.next()

        end = time.time()
        last_idx = len(self.valid_data_loader)

        idx = 0
        while input is not None:
            idx += 1
            last_batch = idx == last_idx

            if self.teacher is not None:
                with torch.no_grad():
                    try:
                        fteacher, logits_teacher = self.teacher(input, is_feat=True)
                    except:
                        logits_teacher = self.teacher(input)
                        fteacher = None
            else:
                fteacher = None
                logits_teacher = None

            # tiny model
            net_result = forward_tiny(model, input)

            # compute loss
            total_loss, reviewkd_loss, dkd_loss = self.compute_loss(net_result, self.criterion, target, use_reviewkd=self.config_dict.use_reviewkd, review_kd_loss_weight=self.config_dict.review_kd_loss_weight, logits_teacher=logits_teacher, fteacher=fteacher, reviewkd_active=True, use_dkd=self.config_dict.use_dkd, dkd_active=True)

            if self.config_dict.use_reviewkd:
                _, net_result = net_result

            acc1, acc5 = accuracy(net_result, target, topk=(1, 5))

            # expanded model
            # net_result = model(input)
            net_result = forward_expanded(model, input)
            if self.config_dict.use_reviewkd:
                _, net_result = net_result
            expanded_acc1, expanded_acc5 = accuracy(net_result, target, topk=(1, 5))

            if not self.config_dict.distributed:
                losses_m.update(total_loss.item(), input.size(0))
                acc1_m.update(acc1.item(), input.size(0))
                acc5_m.update(acc5.item(), input.size(0))
                expanded_acc1_m.update(expanded_acc1.item(), input.size(0))
                expanded_acc5_m.update(expanded_acc5.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time_m.update(time.time() - end)
            if self.config_dict.distributed:
                reduced_loss = reduce_tensor(total_loss.data, self.config_dict.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
                reduced_acc1 = reduce_tensor(acc1.data, self.config_dict.world_size)
                acc1_m.update(reduced_acc1.item(), input.size(0))
                reduced_acc5 = reduce_tensor(acc5.data, self.config_dict.world_size)
                acc5_m.update(reduced_acc5.item(), input.size(0))
                reduced_expanded_acc1 = reduce_tensor(expanded_acc1.data, self.config_dict.world_size)
                expanded_acc1_m.update(reduced_expanded_acc1.item(), input.size(0))
                reduced_expanded_acc5 = reduce_tensor(expanded_acc5.data, self.config_dict.world_size)
                expanded_acc5_m.update(reduced_expanded_acc5.item(), input.size(0))

            if last_batch or idx % self.config_dict.log_interval == 0:
                if self.config_dict.rank == 0:
                    log_name = 'Test' + log_suffix
                    self._logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})  '
                        'Acc5: {acc5.val:>7.4f} ({acc5.avg:>7.4f})  '
                        'E_Acc: {expanded_acc.val:>7.4f} ({expanded_acc.avg:>7.4f})  '
                        'E_Acc5: {expanded_acc5.val:>7.4f} ({expanded_acc5.avg:>7.4f})'.format(
                            log_name, idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, acc=acc1_m, acc5=acc5_m, expanded_acc=expanded_acc1_m, expanded_acc5=expanded_acc5_m))

            end = time.time()
            input, target = prefetcher.next()
            # end for
        if self.config_dict.rank == 0:
            self._logger.info(f"[Validation] Loss:{losses_m.avg:.2f}, Acc:{acc1_m.avg:.2f}%, Acc5:{acc5_m.avg:.2f}%")
        return OrderedDict([('loss', losses_m.avg), ('acc', acc1_m.avg), ('e_acc', expanded_acc1_m.avg), ('acc5', acc5_m.avg), ('e_acc5', expanded_acc5_m.avg)])


    def train(self):
        best_metric = None
        self._logger.info(f'Training data size: {len(self.data_loader)}; Validation data size: {len(self.valid_data_loader)}')
        if len(self.data_loader) < 1:
            self._logger.info('data not enough, return !')
            return

        if self.valid_data_loader is not None:
            self._logger.info("Testing the model on validation set before training")
            eval_metrics = self.validate_es(self.model) if self.config_dict.gpd_ratio > 1 else self.validate(self.model)
            self._logger.info(eval_metrics)

        # if self.lr_scheduler is not None and self.start_epoch > 0:
        #     for i in range(self.start_epoch):
        #         self.lr_scheduler.step()

        if self.config_dict.adaptive_lr:
            self.adaptive_lr_based_on_loss()

        train_metrics = None
        for epoch in range(self.start_epoch, self.config_dict.epoch_init):
            self.data_sampler.set_epoch(epoch)
            # training
            while not isinstance(train_metrics, OrderedDict):
                train_metrics = self.train_one_epoch_es(epoch)
            # self.lr_scheduler.step()
            # validation
            eval_metrics = train_metrics
            if self.valid_data_loader is not None:
                eval_metrics = self.validate_es(self.model) if self.config_dict.gpd_ratio > 1 else self.validate(self.model)
                if self.model_ema is not None and not self.config_dict.model_ema_force_cpu:
                    ema_eval_metrics = self.validate_es(self.model_ema.module, log_suffix=' (EMA)') if self.config_dict.gpd_ratio > 1 else self.validate(self.model_ema.module, log_suffix=' (EMA)')
                    eval_metrics['ema_acc'] = ema_eval_metrics['acc']
                    if math.isnan(ema_eval_metrics['loss']):
                        # self.model_ema = self.build_ema(self.model.module)
                        checkpoint_path = os.path.join(self.config_dict.checkpoint_dir, 'last.pth.tar')
                        load_checkpoint(self.model_ema.module, checkpoint_path, use_ema=False)

            # save checkpoint
            if self.saver is not None:
                # save proper checkpoint with eval metric
                save_metric = max(eval_metrics[self.config_dict.eval_metric], eval_metrics['ema_acc']) if 'ema_acc' in eval_metrics.keys() else eval_metrics[self.config_dict.eval_metric]
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=save_metric)
                self.saver.save_checkpoint_files()

            if self.config_dict.rank == 0:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(self.config_dict.checkpoint_dir, 'summary.txt'),
                    write_header=best_metric is None)
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train_{k}', v, epoch)
                for k, v in eval_metrics.items():
                    self.writer.add_scalar(f'test_{k}', v, epoch)

            train_metrics = None

        if best_metric is not None and self.config_dict.rank == 0:
            _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def main():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='Model Training')
    # args for ddp
    parser.add_argument("--local_rank", default=0, type=int)
    # args for training
    parser.add_argument('--batch_size',  dest='batch_size',  type=int, default=512, help='# images in batch')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default="./check_point/", type=str)
    parser.add_argument('--tensorboard_dir', type=str, default='')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.2, help='learning_rate')
    parser.add_argument('--lr_adjust_step', default=[30, 60, 90], type=int, nargs='+',
                        help='initial learning rate')
    parser.add_argument('--train_data_dir',        dest='train_data_dir', default="/path/to/imagenet", type=str)
    parser.add_argument('--net_version',       dest='model', default='resnet18', help='net_version')
    parser.add_argument('--epoch_init',        dest='epoch_init',  type=int, default=100, help='# epoch_num')
    parser.add_argument('--world_size',           dest='world_size',     type=int, default=-1, help='Number of node in the job')
    parser.add_argument('--rank',                 dest='rank',     type=int, default=-1, help='Rank of the current process')
    parser.add_argument('--load_model_file_path', dest='ckpt_load', type=str, default=None, help='models are loaded from here')
    # other args
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint-hist', type=int, default=5, metavar='N',
                        help='number of checkpoints to keep (default: 10)')
    parser.add_argument('-j', '--workers', type=int, default=16, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no-resume-opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--data_split_stride', type=int, default=-1, help='data stride to sample for validation')
    parser.add_argument('--eval-metric', default='acc', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "acc"')
    # for GPD
    parser.add_argument('--tiny_net_version', dest='tiny_model', default='UNET_v7_3_prelu', help='net_version')
    parser.add_argument('--load_tiny_model_file_path', dest='tiny_ckpt_load', type=str, default=None, help='tiny models are loaded from here')
    parser.add_argument('--tiny_eval_model_ema', action='store_true', help='tiny_eval_model_ema')
    parser.add_argument('--gpd_ratio', type=int, default=1, help='GPD ratio')
    parser.add_argument('--kd_ratio', type=float, default=1, help='weight for kd loss')
    parser.add_argument('--tiny_loss_weight', type=float, default=1, help='weight for tiny ce loss')
    parser.add_argument('--expanded_loss_weight', type=float, default=1, help='weight for dynamic teacher loss')
    parser.add_argument('--kd_loss_type', type=str, default='ce', help='kd loss type, choice: mse | l1')
    parser.add_argument('--skip_es', action='store_true', help='skip_es')
    parser.add_argument('--use_orepa', action='store_true', help='use_orepa')
    parser.add_argument('--deactive_kd_static_teacher', action='store_true', help='deactive_kd_static_teacher')
    parser.add_argument('--adaptive_lr', action='store_true', help='adaptive_lr')
    parser.add_argument('--skip_expanded', action='store_true', help='skip_expanded')
    # for ReviewKD
    parser.add_argument('--use_reviewkd', action='store_true', help='use_reviewkd')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model')
    parser.add_argument('--teacher-weight', type=str, default='torchvision',
                        help='teacher model weight path')
    parser.add_argument('--review-kd-loss-weight', type=float, default=1.0,
                        help='feature konwledge distillation loss weight')
    parser.add_argument('--num_branches_orepa', type=int, default=6, help='num_branches_orepa')

    # for DKD
    parser.add_argument('--use_dkd', action='store_true', help='use_dkd')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                        help='decay factor for model weights moving average (default: 0.99996)')
    parser.add_argument('--eval_model_ema', action='store_true', help='eval_model_ema')
    parser.add_argument('--not_load_ema', action='store_true')

    # data loader
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=5, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    # params for clasification
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float,
                        help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float,
                        help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float,
                        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)",
                        dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str,
                        help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")

    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-large-lr', type=float, default=0, metavar='NORM',
                        help='Clip gradient norm only for those epochs with lr >= clip_large_lr (default: 0)')
    parser.add_argument("--warmup-es-epoch", default=0, type=int, help="warmup_es_epoch")
    parser.add_argument('--first-epoch-lr-ratio', type=float, default=1.0, help='first_epoch_lr_ratio')


    def _parse_args():
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)

        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text

    args, args_text = _parse_args()
    try:
        _ = get_outdir(args.checkpoint_dir, './')
    except:
        print('fail to create dir')

    # save yaml
    with open(os.path.join(args.checkpoint_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # create logger
    log_format = '%(asctime)s %(message)s'
    dist_rank = dist.get_rank() if myutils.is_dist_avail_and_initialized() else 0
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False
    # create console handlers for master process
    if dist_rank == 0:
        print(f'setting up console logger {dist_rank}')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S'))
        _logger.addHandler(console_handler)
    # create file handlers
    print(f'setting up file logger {dist_rank}')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(file_handler)

    _logger.info(args)

    # resolve AMP arguments based on PyTorch / Apex availability
    args.use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        args.use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        args.use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")


    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    args.tiny_model = args.model
    args.batch_size = int(args.batch_size / args.world_size)
    net = Trainer(args, _logger)
    net.train()
    _logger.info("finish training")


if __name__ == "__main__":
    main()

