
import os
import time
import sys
import math
import glob
import yaml

from timm.utils import *
from timm.models import resume_checkpoint, load_checkpoint, model_parameters
from torch.utils.data import Dataset, DataLoader, distributed
from torch.utils.data.dataset import Subset
import torch.distributed as dist
import logging
import myutils
from contextlib import suppress
import datetime
from collections import OrderedDict

import argparse
import h5py
# from vgg_loss import *
# from discriminator_arch import *

# import moxing.pytorch as mox
# from moxing.pytorch.utils.hyper_param_flags import get_flag

from bisect import bisect_left
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import presets
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate
from models import *


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


def init_expand_module(m, new_m):
    # old config
    old_outplane, old_inplane = m.weight.shape[0], m.weight.shape[1]
    old_bias = m.bias
    # new config
    new_outplane, new_inplane = new_m.weight.shape[0], new_m.weight.shape[1]
    new_bias = new_m.bias
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight / expand_outplane_ratio for i in range(expand_outplane_ratio)], 0)
    if expand_inplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_inplane_ratio)], 1)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        assert expand_outplane_ratio * old_bias.shape[0] == new_bias.shape[
            0], f'old_bias {old_bias.shape[0]} has different shape with new_bias{new_bias.shape[0]}'
        new_bias = m.bias.data
        if expand_outplane_ratio > 1:
            new_bias = torch.cat([new_bias / expand_outplane_ratio for i in range(expand_outplane_ratio)], 0)
        new_m.bias.data.copy_(new_bias)


def init_expand_bn(m, new_m):
    # old config
    old_outplane = m.weight.shape[0]
    # new config
    new_outplane = new_m.weight.shape[0]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight / expand_outplane_ratio for i in range(expand_outplane_ratio)], 0)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    new_bias = m.bias.data
    if expand_outplane_ratio > 1:
        new_bias = torch.cat([new_bias / expand_outplane_ratio for i in range(expand_outplane_ratio)], 0)
    new_m.bias.data.copy_(new_bias)
    # init new running_mean
    new_running_mean = m.running_mean.data
    if expand_outplane_ratio > 1:
        new_running_mean = torch.cat([new_running_mean / expand_outplane_ratio for i in range(expand_outplane_ratio)],
                                     0)
    new_m.running_mean.data.copy_(new_running_mean)
    # init new running_mean
    new_running_var = m.running_var.data
    if expand_outplane_ratio > 1:
        new_running_var = torch.cat(
            [new_running_var / expand_outplane_ratio ** 2 for i in range(expand_outplane_ratio)], 0)
    new_m.running_var.data.copy_(new_running_var)


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

class Heaviside(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        i = input.clone()
        ge = torch.ge(i, 0).float()
        step_func = i + (ge.data - i.data)
        return step_func

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.zeros_like(input)
        one = torch.ones_like(input)
        grad = torch.max(zero, torch.abs(one - input)) * grad_input
        return grad


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
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

    torchvision.models.resnet18

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
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


class Trainer():

    def __init__(self, args, _logger):
        self.config_dict = args
        self._logger = _logger
        name_model = eval(args.model)# self.GetNameModel(args['model'])
        object_model = name_model()
        if args.local_rank == 0:
            self._logger.info('Model %s created, param count: %d' %
                         (args.model, sum([m.numel() for m in object_model.parameters()])))
            self.writer = SummaryWriter(self.config_dict.tensorboard_dir if self.config_dict.tensorboard_dir else self.config_dict.output)
        # move model to GPU, enable channels last layout if set
        object_model.cuda()
        if args.channels_last:
            object_model = object_model.to(memory_format=torch.channels_last)

        # create optimizer
        self.optimizer = self.build_optimizer(args, object_model)

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        object_model, self.optimizer, self.amp_autocast, self.loss_scaler = self.setup_amp(object_model, self.optimizer)

        # load pretrained model
        object_model, self.start_epoch = self.load_model(object_model, self.optimizer, self.loss_scaler)

        # setup exponential moving average of model weights, SWA could be used here too
        self.model_ema = self.build_ema(object_model)

        # setup scheduler
        args.lr_scheduler = args.lr_scheduler.lower()
        self.config_dict.lr_scheduler = args.lr_scheduler
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_size,
                                                                gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epoch_init - args.lr_warmup_epochs, eta_min=args.lr_min
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
                )
            self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            self.lr_scheduler = main_lr_scheduler

        # setup distributed training
        self.model = self.setup_ddp(object_model)
        self.saver = self.build_saver(self.model, self.model_ema, self.optimizer)

        # build dataset
        train_dir = os.path.join(args.train_data_dir, "train")
        val_dir = os.path.join(args.train_data_dir, "val")
        dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
        self.data_sampler = train_sampler
        num_classes = len(dataset.classes)
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_categories=num_classes, use_v2=args.use_v2
        )
        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))
        else:
            collate_fn = default_collate

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=self.data_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self.valid_data_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
        )
        self.batch_cnt=len(self.data_loader)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()

    def load_model(self, model, optimizer, loss_scaler=None):
        resume_epoch = 0
        checkpoint_path = os.path.join(self.config_dict.output, 'last.pth.tar')
        if os.path.exists(checkpoint_path):
            self.config_dict.resume = checkpoint_path
            resume_epoch = resume_checkpoint(
                model, self.config_dict.resume,
                optimizer=None if self.config_dict.no_resume_opt else optimizer,
                loss_scaler=None if self.config_dict.no_resume_opt or not self.config_dict.amp else loss_scaler,
                log_info=self.config_dict.local_rank == 0)
            self._logger.info(f'Resume training from epoch {resume_epoch}')
        else:
            ckpt_load = self.config_dict.ckpt_load
            if ckpt_load:
                self._logger.info(f'Load pretrained model from {ckpt_load}')
                load_checkpoint(model, ckpt_load, use_ema=self.config_dict.eval_model_ema)
            else:
                self._logger.info("Train from scratch")
        return model, resume_epoch

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

    def build_ema(self, object_model):
        model_ema = None
        if self.config_dict.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEmaV2(
                object_model, decay=self.config_dict.model_ema_decay, device='cpu' if self.config_dict.model_ema_force_cpu else None)
            if not self.config_dict.resume and self.config_dict.ckpt_load is not None:
                load_checkpoint(model_ema.module, self.config_dict.ckpt_load, use_ema=self.config_dict.eval_model_ema)
            if self.config_dict.resume and not self.config_dict.not_load_ema:
                load_checkpoint(model_ema.module, self.config_dict.resume, use_ema=True)
        return model_ema

    def setup_amp(self, model, optimizer):
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if self.config_dict.use_amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            loss_scaler = ApexScaler()
            if self.config_dict.local_rank == 0:
                self._logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif self.config_dict.use_amp == 'native':
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()
            if self.config_dict.local_rank == 0:
                self._logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            if self.config_dict.local_rank == 0:
                self._logger.info('AMP not enabled. Training in float32.')
        return model, optimizer, amp_autocast, loss_scaler

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
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.config_dict.local_rank])  # can use device str in Torch >= 1.1
            # NOTE: EMA model does not need to be wrapped by DDP
        return model

    def build_saver(self, model, model_ema, optimizer):
        # setup checkpoint saver
        saver = None
        eval_metric = self.config_dict.eval_metric
        if self.config_dict.rank == 0:
            output_base = self.config_dict.output if self.config_dict.output else './output'
            output_dir = output_base
            code_dir = get_outdir(output_dir, 'code')
            myutils.mycopy_tree(os.getcwd(), code_dir)
            decreasing = True if eval_metric == 'loss' else False
            saver = myutils.MyCheckpointSaver(
                model=model, optimizer=optimizer, args=self.config_dict, model_ema=model_ema, amp_scaler=None,
                checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing,
                max_history=self.config_dict.checkpoint_hist)
            if self.config_dict.resume:
                saver.load_checkpoint_files()
        return saver

    def adjust_lr(self, optimizer, idx, epoch, total_epoch, min_lr=2e-5, max_lr=3e-4, step_size=10):
        def np_cyclic_learning_rate(step, lr, max_lr, step_size):
            cycle = math.floor(1. + step / (2. * step_size))
            x     = math.fabs(step / step_size - 2. * cycle + 1.)
            clr   = (max_lr - lr) * max(0., 1. - x)
            return clr + lr
        
        cycle_lr_epoch_num = int(total_epoch*3/5)
        batch_cnt = self.batch_cnt
        cur_epoch = epoch + idx / batch_cnt
        cyclic_lr = np_cyclic_learning_rate(cur_epoch, min_lr, max_lr, step_size) if cur_epoch < cycle_lr_epoch_num else min_lr
        # cyclic_lr = 2e-6
        for param_group in optimizer.param_groups:
            param_group['lr'] = cyclic_lr

    def piecewise_lr(self, optimizer, epoch, th_epoch, max_lr=3e-4, min_lr=3e-5):
        if epoch < th_epoch:
            lr = max_lr
        else:
            lr = min_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self, epoch):
        second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()

        self.model.train()

        end = time.time()
        last_idx = len(self.data_loader) - 1
        num_updates = epoch * len(self.data_loader)
        for batch_idx, (input, target) in enumerate(self.data_loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            input, target = input.cuda(), target.cuda()

            MB = 1024.0 * 1024.0

            with self.amp_autocast():
                outputs = self.model(input)
                loss = self.criterion(outputs, target)

            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            if not self.config_dict.distributed:
                losses_m.update(loss.item(), input.size(0))
                top1_m.update(acc1.item(), input.size(0))

            self.optimizer.zero_grad()
            if self.loss_scaler is not None:
                self.loss_scaler(
                    loss, self.optimizer,
                    clip_grad=self.config_dict.clip_grad, clip_mode=self.config_dict.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in self.config_dict.clip_mode),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                self.optimizer.step()

            if self.model_ema is not None:
                self.model_ema.update(self.model)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % self.config_dict.log_interval == 0:
                lrl = [param_group['lr'] for param_group in self.optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if self.config_dict.distributed:
                    reduced_loss = reduce_tensor(loss.data, self.config_dict.world_size)
                    losses_m.update(reduced_loss.item(), input.size(0))
                    acc1 = reduce_tensor(acc1, self.config_dict.world_size)
                    top1_m.update(acc1.item(), input.size(0))

                if self.config_dict.local_rank == 0:
                    eta_string = str(datetime.timedelta(seconds=int(batch_time_m.avg * (len(self.data_loader) - batch_idx))))
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)] eta: {}   '
                        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                        'Acc: {acc.val:>9.6f} ({acc.avg:>6.4f})  '
                        'Time: {batch_time.val:.3f}s,  '
                        '({batch_time.avg:.3f}s,  '
                        'Mem: {mem:.3e}  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(self.data_loader),
                            100. * batch_idx / last_idx,
                            eta_string,
                            loss=losses_m,
                            acc=top1_m,
                            batch_time=batch_time_m,
                            mem=torch.cuda.max_memory_allocated() / MB,
                            lr=lr,
                            data_time=data_time_m))

            end = time.time()
            # end for

        if hasattr(self.optimizer, 'sync_lookahead'):
            self.optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])


    @torch.no_grad()
    def validate(self, model, log_suffix=''):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        model.eval()

        end = time.time()
        last_idx = len(self.valid_data_loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.valid_data_loader):
                last_batch = batch_idx == last_idx
                input = input.cuda()
                target = target.cuda()
                if self.config_dict.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with self.amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                    if self.config_dict.cls_weight == 0:
                        output = output[1].mean(1)

                loss = self.criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                if self.config_dict.distributed:
                    reduced_loss = reduce_tensor(loss.data, self.config_dict.world_size)
                    acc1 = reduce_tensor(acc1, self.config_dict.world_size)
                    acc5 = reduce_tensor(acc5, self.config_dict.world_size)
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - end)
                end = time.time()
                if self.config_dict.local_rank == 0 and (last_batch or batch_idx % self.config_dict.log_interval == 0):
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))

        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        if self.config_dict.rank == 0:
            _logger.info(f"[Validation] Top1:{top1_m.avg:.2f}%, Top5:{top5_m.avg:.2f}%")
        return metrics


    def train(self):
        best_metric = None
        best_epoch = 0
        self._logger.info(f'Training data size: {len(self.data_loader)}; Validation data size: {len(self.valid_data_loader)}')
        if len(self.data_loader) < 1:
            self._logger.info('data not enough, return !')
            return

        if (self.config_dict.resume or self.config_dict.ckpt_load) and self.valid_data_loader is not None:
            self._logger.info("Testing the pretrained model on validation set before training")
            eval_metrics = self.validate(self.model)
            self._logger.info(eval_metrics)

        for epoch in range(self.start_epoch, self.config_dict.epoch_init):
            if self.config_dict.distributed:
                self.data_sampler.set_epoch(epoch)
            # training
            train_metrics = self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            # validation
            eval_metrics = train_metrics
            if self.valid_data_loader is not None:
                eval_metrics = self.validate(self.model)
                if self.model_ema is not None and not self.config_dict.model_ema_force_cpu:
                    ema_eval_metrics = self.validate(self.model_ema.module, log_suffix=' (EMA)')
                    eval_metrics['ema_top1'] = ema_eval_metrics['top1']

            # save checkpoint
            if self.saver is not None:
                # save proper checkpoint with eval metric
                save_metric = max(eval_metrics[self.config_dict.eval_metric], eval_metrics['ema_top1']) if 'ema_top1' in eval_metrics.keys() else eval_metrics[self.config_dict.eval_metric]
                best_metric, best_epoch = self.saver.save_checkpoint(epoch, metric=save_metric)
                self.saver.save_checkpoint_files()

            if self.config_dict.rank == 0:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(self.config_dict.output, 'summary.txt'),
                    write_header=best_metric is None)
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f'train_{k}', v, epoch)
                for k, v in eval_metrics.items():
                    self.writer.add_scalar(f'test_{k}', v, epoch)

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
    parser.add_argument('--iter_num',    dest='iter_num',    type=int)
    parser.add_argument('--batch_size',  dest='batch_size',  type=int, default=256, help='# images in batch')
    parser.add_argument('--input_size',  dest='input_size',  type=int, default=96, help='# images in batch')
    parser.add_argument('--output_size', dest='output_size', type=int, default=96, help='# images in batch')
    parser.add_argument('--input_dim',   dest='input_dim',   type=int, default=32, help='# input_dim')
    parser.add_argument('--const_dim',   dest='const_dim',   type=int, default=10, help='# input_dim')
    parser.add_argument('--out_dim',     dest='out_dim',     type=int, default=3,  help='# out_dim')
    parser.add_argument('--checkpoint_dir_s3', dest='checkpoint_dir_s3', default='s3://bucket-1762/c00416589/01_MEF/03_data/04_ckpt/MEF/MEF_Pytorch0410/', type=str)
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default="/cache/c00416589/MEF/check_point/", type=str)
    parser.add_argument('--output', dest='output', default="./", type=str)
    parser.add_argument('--tensorboard_dir', type=str, default='')
    parser.add_argument('--train_data_dir',        dest='train_data_dir', default="/cache/c00416589/MEF/data/", type=str)
    parser.add_argument('--valid_data_dir',        dest='valid_data_dir', default=None, type=str)
    parser.add_argument('--num_gpus',          dest='num_gpus', default='1', help='# num_gpus')
    parser.add_argument('--inv_vst',           dest='inv_vst', type=int, default=0, help='# inv_vst')
    parser.add_argument('--pca',               dest='pca', type=int, default=1, help='# calculate asymloss only using Y channel')
    parser.add_argument('--noise_newloss',     dest='noise_newloss', type=int, default=0, help='# noise_newloss')
    parser.add_argument('--net_version',       dest='model', default='UNET_v7_3_prelu', help='net_version')
    parser.add_argument('--gpu_idx',           dest='gpu_idx',     default='0,1,2,3,4,5,6,7', help='# gpu idx')
    parser.add_argument('--epoch_init',        dest='epoch_init',  type=int, default=90, help='# epoch_num')
    parser.add_argument('--shuffle',           dest='shuffle',     type=int, default=0, help='# data shuffle or not')
    parser.add_argument('--init_method',          dest='init_method',     type=str, default='tcp://127.0.0.1:12345', help='URL specifying how to initialize the process group')
    parser.add_argument('--world_size',           dest='world_size',     type=int, default=-1, help='Number of node in the job')
    parser.add_argument('--rank',                 dest='rank',     type=int, default=-1, help='Rank of the current process')
    parser.add_argument('--load_model_file_path', dest='ckpt_load', type=str, default=None, help='models are loaded from here')
    parser.add_argument('--train_data_name', dest='train_data_name', type=str, default='*.h5', help='train data name')
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
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                        help='decay factor for model weights moving average (default: 0.99996)')
    parser.add_argument('--eval_model_ema', action='store_true', help='eval_model_ema')
    parser.add_argument('--not_load_ema', action='store_true')
    # params for clasification
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.1, help='learning_rate')
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters of all layers (default: None, same value as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
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
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")


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
    args.output = args.checkpoint_dir
    try:
        _ = get_outdir(args.output, './')
        # Path(os.path.join(args.output, 'occlusion')).mkdir(parents=True, exist_ok=True)
    except:
        print('fail to create dir')

    # save yaml
    with open(os.path.join(args.output, 'args.yaml'), 'w') as f:
        f.write(args_text)

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
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
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
    file_handler = logging.FileHandler(os.path.join(args.output, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S'))
    _logger.addHandler(file_handler)


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

    args.batch_size = int(args.batch_size / args.world_size)
    net = Trainer(args, _logger)
    net.train()
    _logger.info("finish training")


if __name__ == "__main__":
    main()

