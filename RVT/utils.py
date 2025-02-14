# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time
from collections import defaultdict, deque
import datetime
import cv2
from PIL import Image

import torch
import torch.distributed as dist
import numpy as np
from torchvision import datasets, transforms

import cifar_augmentations
import imagenet_augmentations
from torch.autograd import Variable
import torch.nn as nn

import torch.nn.functional as F
from robust_models import Attention, MatchedDropout, MatchedDropPath
from timm.utils import *


data_loaders_names = {
    'Brightness': 'brightness',
    'Contrast': 'contrast',
    'Defocus Blur': 'defocus_blur',
    'Elastic Transform': 'elastic_transform',
    'Fog': 'fog',
    'Frost': 'frost',
    'Gaussian Noise': 'gaussian_noise',
    'Glass Blur': 'glass_blur',
    'Impulse Noise': 'impulse_noise',
    'JPEG Compression': 'jpeg_compression',
    'Motion Blur': 'motion_blur',
    'Pixelate': 'pixelate',
    'Shot Noise': 'shot_noise',
    'Snow': 'snow',
    'Zoom Blur': 'zoom_blur'
}

def get_ce_alexnet():
    """Returns Corruption Error values for AlexNet"""

    ce_alexnet = dict()
    ce_alexnet['Gaussian Noise'] = 0.886428
    ce_alexnet['Shot Noise'] = 0.894468
    ce_alexnet['Impulse Noise'] = 0.922640
    ce_alexnet['Defocus Blur'] = 0.819880
    ce_alexnet['Glass Blur'] = 0.826268
    ce_alexnet['Motion Blur'] = 0.785948
    ce_alexnet['Zoom Blur'] = 0.798360
    ce_alexnet['Snow'] = 0.866816
    ce_alexnet['Frost'] = 0.826572
    ce_alexnet['Fog'] = 0.819324
    ce_alexnet['Brightness'] = 0.564592
    ce_alexnet['Contrast'] = 0.853204
    ce_alexnet['Elastic Transform'] = 0.646056
    ce_alexnet['Pixelate'] = 0.717840
    ce_alexnet['JPEG Compression'] = 0.606500

    return ce_alexnet

def get_mce_from_accuracy(accuracy, error_alexnet):
    """Computes mean Corruption Error from accuracy"""
    error = 100. - accuracy
    ce = error / (error_alexnet * 100.)

    return ce

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.2f} ({global_avg:.2f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, logger, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        try:
            space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        except:
            space_fmt = ':' + str(len(str(len([1])))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def aug(args, image, preprocess):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    if 'cifar' in args.data_set:
        aug_list = cifar_augmentations.augmentations
        if args.all_ops:
            aug_list = cifar_augmentations.augmentations_all
    else:
        aug_list = imagenet_augmentations.augmentations
        if args.all_ops:
            aug_list = imagenet_augmentations.augmentations_all
    ws = np.float32(
        np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
    m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, args, dataset, preprocess):
        self.args = args
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = not args.jsd

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.no_jsd:
            return aug(self.args, x, self.preprocess), y
        else:
            im_tuple = (self.preprocess(x), aug(self.args, x, self.preprocess),
                        aug(self.args, x, self.preprocess))
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def my_add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if 'controller' not in name:
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]



def confidence_score(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = F.softmax(output, 1)
    confidence = output.gather(1, target.unsqueeze(-1))
    return confidence


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def SupConLoss(features, labels=None, mask=None, out_logit=None, temperature=0.07, supcontra=True):
    device = features.device
    # temperature=0.07
    base_temperature=temperature
    batch_size = features.shape[0]//2

    if not supcontra:
        labels = None
        mask = None

    features = F.normalize(features, dim=1)
    f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1) # B x 2 x N

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def SupConLossCopy(features, labels=None, mask=None, out_logit=None, temperature=0.07, supcontra=True):
    device = features.device
    # temperature=0.07
    base_temperature=temperature
    batch_size = features.shape[0]

    if not supcontra:
        labels = torch.cat([torch.arange(batch_size//2) for i in range(2)], dim=0).to(device)

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    features = F.normalize(features, dim=1)
    anchor_feature = features.float()
    contrast_feature = features.float()

    similarity = torch.matmul(anchor_feature, contrast_feature.T)
    cosine_similarity = similarity
    anchor_dot_contrast = torch.div(cosine_similarity, temperature)

    # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    # logits = anchor_dot_contrast - logits_max.detach()

    logits = anchor_dot_contrast

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # print(mask.sum())

    # compute log_prob
    # print(logits)
    # print(torch.exp(logits))
    # assert False
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    if out_logit is not None:
        mean_log_prob_pos = mean_log_prob_pos * (1-out_logit)

    # loss
    mean_log_prob_pos = - (temperature / base_temperature) * mean_log_prob_pos
    mean_log_prob_pos = mean_log_prob_pos.view(batch_size)

    # N_nonzeor = torch.nonzero(mask.sum(1)).shape[0]
    # loss = mean_log_prob_pos.sum()/N_nonzeor

    loss = mean_log_prob_pos.mean()

    if torch.isnan(loss):
        # print("nan contrastive loss")
        loss=torch.zeros(1).to(device)
    return loss


def standard_plot_attn_on_image(original_image, mask, save_path):
    w = int(mask.shape[0]**0.5)
    transformer_attribution = mask.reshape(1,1,w,w)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=224//w, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * transformer_attribution), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(image_transformer_attribution)
    cam = cam / np.max(cam)

    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    # Image.fromarray(vis).save(save_path, quality=85, optimize=True)
    out = Image.fromarray(vis)
    t = transforms.ToTensor()
    out = t(out)
    return out


def normalize_data(x, args):
    if args.data_set == 'CIFAR10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
    elif 'IMNET' in args.data_set:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    mu = torch.tensor(mean).view(3, 1, 1).to(x.device)
    std = torch.tensor(std).view(3, 1, 1).to(x.device)
    return (x - mu)/std


def denormalize_data(x, args):
    if args.data_set == 'CIFAR10':
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
    elif 'IMNET' in args.data_set:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    mu = torch.tensor(mean).view(3, 1, 1).to(x.device)
    std = torch.tensor(std).view(3, 1, 1).to(x.device)
    return (x * std) + mu


def activate_clean_attention_noisy_feature(model):
    for module in model.modules():
        if isinstance(module, (Attention)):
            module.clean_attention_noisy_feature = True


def deactivate_clean_attention_noisy_feature(model):
    for module in model.modules():
        if isinstance(module, (Attention)):
            module.clean_attention_noisy_feature = False


def activate_noisy_attention_clean_feature(model):
    for module in model.modules():
        if isinstance(module, (Attention)):
            module.noisy_attention_clean_feature = True


def deactivate_noisy_attention_clean_feature(model):
    for module in model.modules():
        if isinstance(module, (Attention)):
            module.noisy_attention_clean_feature = False


def set_distraction_loss_type(model, distraction_loss_type):
    for module in model.modules():
        if isinstance(module, (Attention)):
            module.distraction_loss_type = distraction_loss_type


def AttnDistractLoss(model):
    loss = 0
    # anchor = 6
    counter = 0
    final_counter = 1e-4
    for module in model.modules():
        if isinstance(module, (Attention)):
            if module.distraction_loss is not None:
                loss += module.distraction_loss
                final_counter += 1
                # print(counter)
            counter += 1
    loss = loss / final_counter
    return loss




def get_attn(model):
    attn = []
    for module in model.modules():
        if isinstance(module, (Attention)):
            if module.vis_attn is not None:
                attn.append(module.vis_attn.clone())
    return attn


'''
@Parameter atten_grad, ce_grad: should be 2D tensor with shape [batch_size, -1]
'''
def PCGrad(atten_grad, ce_grad, sim, shape):
    pcgrad = atten_grad[sim < 0]
    temp_ce_grad = ce_grad[sim < 0]
    dot_prod = torch.mul(pcgrad, temp_ce_grad).sum(dim=-1)
    dot_prod = dot_prod / torch.norm(temp_ce_grad, dim=-1)
    pcgrad = pcgrad - dot_prod.view(-1, 1) * temp_ce_grad
    atten_grad[sim < 0] = pcgrad
    atten_grad = atten_grad.view(shape)
    return atten_grad


def set_matcheddrop_mode(model, mode):
    for module in model.modules():
        if isinstance(module, (MatchedDropout, MatchedDropPath)):
            module.mode = mode


class MyCheckpointSaver(CheckpointSaver):
    def __init__(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):

        super().__init__(model,
                         optimizer,
                         args,
                         model_ema,
                         amp_scaler,
                         checkpoint_prefix,
                         recovery_prefix,
                         checkpoint_dir,
                         recovery_dir,
                         decreasing,
                         max_history,
                         unwrap_fn)

    def load_checkpoint_files(self):
        tmp_save_path = os.path.join(self.checkpoint_dir, 'checkpoint_files' + self.extension)
        file_to_load = torch.load(tmp_save_path)
        self.checkpoint_files = file_to_load['checkpoint_files']
        self.best_metric = file_to_load['best_metric']

    def save_checkpoint_files(self):
        tmp_save_path = os.path.join(self.checkpoint_dir, 'checkpoint_files' + self.extension)
        file_to_save = {'checkpoint_files': self.checkpoint_files, 'best_metric': self.best_metric}
        torch.save(file_to_save, tmp_save_path)




def get_numclasses_stats(args):
    if args.data_set == 'IMNET':
        nb_classes = 1000
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif args.data_set == 'IMNET100':
        nb_classes = 100
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif args.data_set == 'CIFAR10':
        nb_classes = 10
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
    elif args.data_set == 'CIFAR100':
        nb_classes = 100
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    return nb_classes, mean, std


def SwitchAttentionLoss(model):
    loss = 0
    counter = 0
    final_counter = 1e-4
    for module in model.modules():
        if isinstance(module, (Attention)):
            if module.attn_loss is not None:
                loss += module.attn_loss
                final_counter += 1
            counter += 1
    loss = loss / final_counter
    return loss


def set_attn_loss_type(model, attn_loss_type):
    for module in model.modules():
        if isinstance(module, Attention):
            module.attn_loss_type = attn_loss_type


