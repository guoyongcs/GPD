import os
import time
import torch
import torch.nn as nn
import argparse
import logging
import yaml
from collections import OrderedDict
import sys
from train_dkd import Trainer, data_prefetcher
import myutils
from timm.utils import *
from timm.models import load_checkpoint
import torch.distributed as dist
import numpy as np

from orepa_ft import transfer2orepa, OREPA
from gpd_utils import gpd_init, forward_tiny, forward_expanded, extract_tiny_net
from models import *


_logger = logging.getLogger('test')

class Tester(Trainer):
    def __init__(self, config_dict_, _logger):

        self.config_dict = config_dict_
        self._logger = _logger
        name_model = eval(self.config_dict.model)  # self.GetNameModel(args['model'])
        object_model = name_model()
        self.object_model = object_model
        if self.config_dict.local_rank == 0:
            self._logger.info('Model %s created, param count: %d' %
                         (name_model.__name__, sum([m.numel() for m in object_model.parameters()])))

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

        # load pretrainded expanded model
        if self.config_dict.ckpt_load:
            self._logger.info(f"Loading pre-trained checkpoint from {self.config_dict.ckpt_load}...")
            load_checkpoint(object_model, self.config_dict.ckpt_load, use_ema=self.config_dict.eval_model_ema)

        # move model to GPU, enable channels last layout if set
        object_model.cuda()
        if self.config_dict.channels_last:
            object_model = object_model.to(memory_format=torch.channels_last)

        # # setup exponential moving average of model weights, SWA could be used here too
        # self.model_ema = self.build_ema(object_model)
        self.model = object_model

        self.make_dataset_imagenet()
        self.batch_cnt = len(self.data_loader)


    @torch.no_grad()
    def validate_es(self, model, log_suffix=''):
        batch_time_m = AverageMeter()
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

            # tiny model
            net_result = forward_tiny(model, input)
            acc1, acc5 = accuracy(net_result, target, topk=(1, 5))

            # expanded model
            net_result = forward_expanded(model, input)
            if self.config_dict.use_reviewkd:
                _, net_result = net_result
            expanded_acc1, expanded_acc5 = accuracy(net_result, target, topk=(1, 5))

            if not self.config_dict.distributed:
                acc1_m.update(acc1.item(), input.size(0))
                acc5_m.update(acc5.item(), input.size(0))
                expanded_acc1_m.update(expanded_acc1.item(), input.size(0))
                expanded_acc5_m.update(expanded_acc5.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time_m.update(time.time() - end)
            if self.config_dict.distributed:
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
                        'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})  '
                        'Acc5: {acc5.val:>7.4f} ({acc5.avg:>7.4f})  '
                        'E_Acc: {expanded_acc.val:>7.4f} ({expanded_acc.avg:>7.4f})  '
                        'E_Acc5: {expanded_acc5.val:>7.4f} ({expanded_acc5.avg:>7.4f})'.format(
                            log_name, idx, last_idx, batch_time=batch_time_m,
                            acc=acc1_m, acc5=acc5_m, expanded_acc=expanded_acc1_m, expanded_acc5=expanded_acc5_m))

            end = time.time()
            input, target = prefetcher.next()
            # end for
        if self.config_dict.rank == 0:
            self._logger.info(f"[Validation] Acc:{acc1_m.avg:.2f}%, Acc5:{acc5_m.avg:.2f}%")
        return OrderedDict([('acc', acc1_m.avg), ('e_acc', expanded_acc1_m.avg), ('acc5', acc5_m.avg), ('e_acc5', expanded_acc5_m.avg)])


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
            net_result = model(input)

            # compute acc
            acc1, acc5 = accuracy(net_result, target, topk=(1, 5))

            if not self.config_dict.distributed:
                acc1_m.update(acc1.item(), input.size(0))

            torch.cuda.synchronize()
            batch_time_m.update(time.time() - end)
            if self.config_dict.distributed:
                reduced_acc1 = reduce_tensor(acc1.data, self.config_dict.world_size)
                acc1_m.update(reduced_acc1.item(), input.size(0))

            if last_batch or idx % self.config_dict.log_interval == 0:
                if self.config_dict.rank == 0:
                    log_name = 'Test' + log_suffix
                    self._logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Acc: {acc.val:>7.4f} ({acc.avg:>7.4f})  '.format(
                            log_name, idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, acc=acc1_m))

            end = time.time()
            input, target = prefetcher.next()
            # end for
        if self.config_dict.rank == 0:
            self._logger.info(f"[Validation] Acc:{acc1_m.avg:.2f}%")
        return OrderedDict([('acc', acc1_m.avg)])


def parse_args():
    config_parser = parser = argparse.ArgumentParser(description='Testing Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='Model Testing')

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_data_dir', dest='train_data_dir', default="/path/to/imagenet", type=str)
    parser.add_argument('--model', dest='model', default='resnet18', help='net_version')
    parser.add_argument('--load_tiny_model_file_path', dest='tiny_ckpt_load', type=str, default=None, help='tiny models are loaded from here')
    parser.add_argument('--skip_es', action='store_true', help='skip_es')
    parser.add_argument('--use_orepa', action='store_true', help='use_orepa')
    parser.add_argument('--num_branches_orepa', type=int, default=6, help='num_branches_orepa')

    parser.add_argument('--load_model_file_path', dest='ckpt_load', type=str, default=None,
                        help='models are loaded from here')

    # For test
    parser.add_argument('--gpd_ratio', type=int, default=1, help='GPD ratio')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval_model_ema', action='store_true', help='eval_model_ema')
    parser.add_argument('--deploy', action='store_true', default=False)
    parser.add_argument('--deploy_model_path', type=str, default=False)

    parser.add_argument('--interpolation', default='bilinear', type=str)
    parser.add_argument('--val-resize-size', default=256, type=int)
    parser.add_argument('--val-crop-size', default=224, type=int)
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument('--workers', type=int, default=4, metavar='N')
    parser.add_argument('--channels-last', action='store_true', default=False)
    parser.add_argument('--use_reviewkd', action='store_true', help='use_reviewkd')

    def _parse_args():
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        args = parser.parse_args(remaining)
        return args, yaml.safe_dump(args.__dict__, default_flow_style=False)

    args, args_text = _parse_args()
    return args, args_text

def setup_env(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Testing in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Testing with a single process on 1 GPU.')


    torch.manual_seed(args.local_rank)
    np.random.seed(args.local_rank)


def setup_logging(args):
    log_format = '%(asctime)s %(message)s'
    dist_rank = dist.get_rank() if myutils.is_dist_avail_and_initialized() else 0
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt='%Y-%m-%d %H:%M:%S'))
        _logger.addHandler(console_handler)

def load_checkpoint(model, ckpt_path, use_ema=False):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if use_ema and 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    model.load_state_dict(state_dict, strict=False)
    return model

def replace_orepa_with_conv(model):
    """
    Completely replace the OREPA layers in the model with Conv2d layers.

    Args:
        model: A PyTorch model containing OREPA layers.

    Returns:
        model: The model with all OREPA layers replaced by Conv2d layers.
    """

    def _replace_module(module, path=''):
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name

            if isinstance(child, OREPA):
                # Get the equivalent convolution parameters from the OREPA layer
                child.eval()
                kernel, bias = child.weight_gen()

                # Create the equivalent Conv2d layer
                conv = nn.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=True
                )

                # Set the parameters of the Conv2d layer
                conv.weight.data = kernel
                conv.bias.data = bias

                # If the OREPA layer has a nonlinear activation function, keep it
                if hasattr(child, 'nonlinear') and not isinstance(child.nonlinear, nn.Identity):
                    # Create a Sequential containing the Conv2d and the nonlinear layer
                    new_module = nn.Sequential(conv, child.nonlinear)
                else:
                    new_module = conv

                # Replace the module in the model
                module_path = child_path.split('.')
                parent = model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], new_module)
            else:
                _replace_module(child, child_path)

    # Replace all OREPA layers
    _replace_module(model)

    return model



def main():
    args, args_text = parse_args()
    setup_env(args)
    setup_logging(args)
    _logger.info(args)

    tester = Tester(args, _logger)
    test_model = tester.model
    if args.deploy:
        test_model = extract_tiny_net(test_model, test_sample=torch.rand(1, 3, 224, 224).cuda())
        # convert orepa to conv
        test_model = replace_orepa_with_conv(test_model)
        # Save the deploy model
        torch.save(test_model.state_dict(), args.deploy_model_path)
        _logger.info(f"The deploy model is saved at {args.deploy_model_path}")

        # Test the deploy model
        deploy_model = tester.object_model
        _logger.info("Loading saved deploy model for testing")
        deploy_model.load_state_dict(torch.load(args.deploy_model_path, map_location='cpu'), strict=False)
        _logger.info("Starting evaluation...")
        eval_metrics = tester.validate(deploy_model.cuda())
    else:
        _logger.info("Starting evaluation...")
        if args.gpd_ratio > 1:
            eval_metrics = tester.validate_es(test_model)
        else:
            eval_metrics = tester.validate(test_model)



if __name__ == '__main__':
    main()