from torch.nn.modules.utils import _ntuple
from typing import Optional, OrderedDict, Tuple, Union, TypeVar
import torch.nn.qat.modules as nnqat
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import torch.nn.intrinsic.modules as nni
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import math
import warnings
from orepa_ft import OREPA, OREPA_SE_4x, transVI_multiscale, merge_conv, transfer2orepa
from models.fan import TokenMixing, ChannelProcessing, FANBlock, ClassAttn
from models.convnext_utils import ConvNeXtBlock
from einops import rearrange
from robust_models import Attention


def get_outchannels(m):
    if hasattr(m, 'out_channels'):
        return 'out_channels'
    elif hasattr(m, 'out_features'):
        return 'out_features'
    elif hasattr(m, 'num_features'):
        return 'num_features'
    elif hasattr(m, 'num_parameters'):
        return 'num_parameters'
    elif hasattr(m, 'dim'):
        return 'dim'
    elif hasattr(m, 'normalized_shape'):
        return 'normalized_shape'
    else:
        return None

def get_inchannels(m):
    if hasattr(m, 'in_channels'):
        return 'in_channels'
    elif hasattr(m, 'in_features'):
        return 'in_features'
    elif hasattr(m, 'num_features'):
        return 'num_features'
    elif hasattr(m, 'num_parameters'):
        return 'num_parameters'
    elif hasattr(m, 'dim'):
        return 'dim'
    elif hasattr(m, 'normalized_shape'):
        return 'normalized_shape'
    else:
        return None


def get_attr_value(x):
    if isinstance(x, tuple):
        v = x[0]
    else:
        v = x
    return v


def check_contain_orepa(source_model):
    contain_orepa = False
    for name_module in source_model.named_modules():
        name, m = name_module
        if isinstance(m, OREPA):
            contain_orepa = True
            break
    return contain_orepa


def export_sequential(model):
    if isinstance(model, nn.Sequential):
        new_model = nn.Sequential(
            *(module.export() if hasattr(module, 'export') else module for module in model)
        )
        return new_model
    else:
        assert False, "It is not nn.Sequential."


def get_same_padding(kernel_size: Union[int, Tuple[int, int]]) -> Union[int, tuple]:
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, f"invalid kernel size: {kernel_size}"
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    else:
        assert isinstance(
            kernel_size, int
        ), "kernel size should be either `int` or `tuple`"
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class DynamicModule(nn.Module):  # important
    def export(self) -> nn.Module:
        raise NotImplementedError

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = self.state_dict()
        for prefix, module in self.named_children():
            if isinstance(module, DynamicModule):
                for name, tensor in module.active_state_dict().items():
                    state_dict[prefix + "." + name] = tensor
        return state_dict


class DynamicConv2d(DynamicModule, nn.Conv2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        gpd_ratio: int = 1,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels  # important
        self.active_out_channels = out_channels  # important
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight, self.active_out_channels, self.active_in_channels)

        if self.groups == 1:
            weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        bias = self.bias[: self.active_out_channels].contiguous()
        # bias = avg_param_for_tiny_model1d(bias, self.bias, self.active_out_channels)
        if self.groups == 1:
            bias = bias * self.active_ratio
        return bias

    @property  # important
    def active_groups(self):
        return max(self.groups // self.active_ratio, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]

        if self.groups > 1:
            x = x * self.gpd_ratio / self.active_ratio

        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            output = getattr(F, "conv{}d".format(self._ndim))(
                x,
                self.active_weight,  # important
                self.active_bias,  # important
                stride=self.stride,
                # padding=get_same_padding(int(active_weight.size(2))) * self.dilation[0],
                padding=self.padding,
                dilation=self.dilation,
                groups=self.active_groups,
            )

            if self.groups > 1:
                output = output / self.gpd_ratio * self.active_ratio

            return output

    def export(self) -> nn.Module:  # important
        module = getattr(nn, "Conv{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            # padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            padding=self.padding,
            dilation=self.dilation,
            groups=self.active_groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            if self.groups > 1:
                scale_factor = self.gpd_ratio / self.active_ratio
                state_dict["bias"] = self.active_bias / scale_factor
            else:
                state_dict["bias"] = self.active_bias
        return state_dict

    def extra_repr(self):
        s = ('{active_in_channels}, {active_out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class DynamicLinear(nn.Linear, DynamicModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, gpd_ratio: int = 1) -> None:
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.active_in_features = in_features  # important
        self.active_out_features = out_features  # important
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        # return self.weight[
        #     : self.active_out_features, : self.active_in_features
        # ].contiguous() * self.active_ratio
        weight = self.weight[: self.active_out_features, : self.active_in_features].contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight, self.active_out_features, self.active_in_features)
        return weight * self.active_ratio

    @property  # important
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        # return self.bias[: self.active_out_features].contiguous() * self.active_ratio
        bias = self.bias[: self.active_out_features].contiguous()
        # bias = avg_param_for_tiny_model1d(bias, self.bias, self.active_out_features)
        return bias * self.active_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_features = x.shape[-1]  # important
        try:
            return F.linear(x, weight=self.active_weight, bias=self.active_bias)  # important
        except:
            assert False, f"Consider putting {self.module_name} into exclude_module_name_list"

    def export(self) -> nn.Module:  # important
        module = nn.Linear(
            self.active_in_features,
            self.active_out_features,
            bias=self.bias is not None,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicConvTranspose2d(DynamicModule, nn.ConvTranspose2d):
    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        output_padding: Union[int, Tuple] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple] = 1,
        gpd_ratio: int = 1,
    ) -> None:
        kernel_size = _ntuple(self._ndim)(kernel_size)
        stride = _ntuple(self._ndim)(stride)
        dilation = _ntuple(self._ndim)(dilation)
        nn.ConvTranspose2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode="zeros",
        )
        self.active_in_channels = in_channels  # important
        self.active_out_channels = out_channels  # important
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_in_channels, : self.active_out_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight, self.active_in_channels, self.active_out_channels)
        if self.groups == 1:
            weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        bias = self.bias[: self.active_out_channels].contiguous()
        # bias = avg_param_for_tiny_model1d(bias, self.bias, self.active_out_channels)
        if self.groups == 1:
            bias = bias * self.active_ratio
        return bias
        # return self.bias[: self.active_out_channels].contiguous() * self.active_ratio

    @property  # important
    def active_groups(self):
        return max(self.groups // self.active_ratio, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.active_in_channels = x.shape[1]  # important
        if self.padding_mode != "zeros":
            raise NotImplementedError
        else:
            try:
                return getattr(F, "conv_transpose{}d".format(self._ndim))(
                    x,
                    self.active_weight,  # important
                    self.active_bias,  # important
                    stride=self.stride,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    groups=self.active_groups,
                    dilation = self.dilation,
                )
            except:
                assert False, f"Consider putting {self.module_name} into exclude_module_name_list"

    def export(self) -> nn.Module:  # important
        module = getattr(nn, "ConvTranspose{}d".format(self._ndim))(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.active_groups,
            bias=self.bias is not None,
            dilation=self.dilation,
            padding_mode=self.padding_mode,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict

    def extra_repr(self):
        s = ('{active_in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class DynamicBatchNorm2d(DynamicModule, nn.BatchNorm2d):
    _ndim = 2

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        gpd_ratio: int = 1,
    ) -> None:
        nn.BatchNorm2d.__init__(
            self,
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.active_num_features = num_features
        self.gpd_ratio = gpd_ratio
        self.active_ratio = 1
        self.min_out_channels = num_features
        # self.register_full_backward_hook(self._backward_grad_scale_hook)
        self.register_buffer('tiny_running_mean', torch.zeros(num_features))
        self.register_buffer('tiny_running_var', torch.ones(num_features))

        # self.weight.register_hook(self._running_stats_calib)

    def _running_stats_calib(self, grad):
        if self.training:
            if self.gpd_ratio > 1 and self.active_ratio == self.gpd_ratio:
                self.running_mean.copy_(self.previous_running_mean)
                self.running_var.copy_(self.previous_running_var)
                self.num_batches_tracked.add_(-1)
        return grad

    @property
    def active_running_mean(self) -> Optional[torch.Tensor]:
        if self.running_mean is None:
            return None
        if self.gpd_ratio > 1 and self.active_ratio == self.gpd_ratio: # tiny model
            return self.tiny_running_mean[: self.active_num_features]
        else:
            return self.running_mean[: self.active_num_features]

    @property
    def active_running_var(self) -> Optional[torch.Tensor]:
        if self.running_var is None:
            return None
        if self.gpd_ratio > 1 and self.active_ratio == self.gpd_ratio: # tiny model
            return self.tiny_running_var[: self.active_num_features]
        else:
            return self.running_var[: self.active_num_features]

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None

        weight = self.weight[: self.active_num_features]
        # weight = avg_param_for_tiny_model1d(weight, self.weight, self.active_num_features)

        return weight

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        bias = self.bias[: self.active_num_features]
        # bias = avg_param_for_tiny_model1d(bias, self.bias, self.active_num_features)
        return bias

    @property
    def active_eps(self):
        return self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        if self.tiny_running_mean.mean() == 0:
            self.tiny_running_mean.copy_(self.running_mean)
            self.tiny_running_var.copy_(self.running_var)

        self.active_num_features = x.shape[1]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True

        else:
            bn_training = (self.active_running_mean is None) and (
                self.active_running_var is None
            )

        # preprocess
        x = x * self.gpd_ratio / self.active_ratio

        output = F.batch_norm(
            x,
            self.active_running_mean,
            self.active_running_var,
            self.active_weight,
            self.active_bias,
            bn_training,
            exponential_average_factor,
            self.active_eps,
        )

        # restore with scale
        output = output / self.gpd_ratio * self.active_ratio

        return output

    def export(self) -> nn.Module:
        scale_factor = self.gpd_ratio / self.active_ratio
        module = getattr(nn, "BatchNorm{}d".format(self._ndim))(
            self.active_num_features,
            eps=self.active_eps / (scale_factor ** 2),
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        module.load_state_dict(self.active_state_dict())

        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        scale_factor = self.gpd_ratio / self.active_ratio
        if self.running_mean is not None:
            state_dict["running_mean"] = self.active_running_mean / scale_factor
        if self.running_var is not None:
            state_dict["running_var"] = self.active_running_var / (scale_factor ** 2)
        if self.weight is not None:
            state_dict["weight"] = self.active_weight / scale_factor
        if self.bias is not None:
            state_dict["bias"] = self.active_bias / scale_factor

        state_dict.pop('tiny_running_mean', None)
        state_dict.pop('tiny_running_var', None)

        return state_dict


class DynamicPReLU(DynamicModule, nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None, gpd_ratio: int = 1) -> None:
        nn.PReLU.__init__(self, num_parameters, init, device, dtype)
        self.active_num_parameters = num_parameters
        self.gpd_ratio = gpd_ratio
        self.active_ratio = 1

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        weight = self.weight[: self.active_num_parameters]
        # weight = avg_param_for_tiny_model1d(weight, self.weight, self.active_num_parameters)
        return weight

    def forward(self, input):
        try:
            return F.prelu(input, self.active_weight)
        except:
            assert False, f"Consider putting {self.module_name} into exclude_module_name_list"

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        state_dict["weight"] = self.active_weight
        return state_dict

    def export(self) -> nn.Module:  # important
        module = nn.PReLU(num_parameters=self.active_num_parameters)
        module.load_state_dict(self.active_state_dict())
        return module


class scale_module(nn.Module):
    def __init__(self, shape, valid_dim):
        super(scale_module, self).__init__()
        self.valid_dim = valid_dim
        self.out_channels = shape[valid_dim]
        self.weight = nn.Parameter(torch.Tensor(shape))

    def forward(self, input):
        output = input * self.weight
        return output


class DynamicScale(DynamicModule, scale_module):
    def __init__(self, shape, valid_dim, gpd_ratio: int = 1) -> None:
        scale_module.__init__(self, shape, valid_dim)
        self.active_out_channels = self.out_channels
        self.gpd_ratio = gpd_ratio
        self.active_ratio = 1

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        output = None
        if self.valid_dim == 0:
            output = self.weight[: self.active_out_channels]
        elif self.valid_dim == 1:
            output = self.weight[:, : self.active_out_channels]
        elif self.valid_dim == 2:
            output = self.weight[:, :, : self.active_out_channels]
        elif self.valid_dim == 3:
            output = self.weight[:, :, :, : self.active_out_channels]
        return output

    def forward(self, input):
        try:
            return input * self.active_weight
        except:
            assert False, f"Consider putting {self.module_name} into exclude_module_name_list"


class _ConvNd(nn.modules.conv._ConvNd):

    MOD = TypeVar('MOD', bound=nn.modules.conv._ConvNd)
    _FLOAT_MODULE = MOD

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 qconfig=None,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, transposed,
                                         output_padding, groups, bias, padding_mode, **factory_kwargs)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)

    @staticmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module

            Args:
               `mod`: a float module, either produced by torch.ao.quantization utilities
               or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__  # type: ignore[attr-defined]
        )
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        if issubclass(type(mod), nni._FusedModule):
            mod = mod[0]  # type: ignore[index]
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, qconfig=qconfig)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv

    def to_float(self):
        """ This works for both single qat conv, and the qat conv - relu modules
        to convert the qat module to a floating point module
        """
        cls = type(self)
        conv = cls._FLOAT_CONV_MODULE(  # type: ignore[attr-defined, operator]
            self.in_channels,
            self.out_channels,
            self.kernel_size,  # type: ignore[arg-type]
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.dilation,  # type: ignore[arg-type]
            self.groups,
            self.bias is not None,
            self.padding_mode)
        conv.weight = torch.nn.Parameter(self.weight.detach())
        if self.bias is not None:
            conv.bias = torch.nn.Parameter(self.bias.detach())
        # conv relu
        if issubclass(cls, nni._FusedModule):
            modules = [conv]
            assert hasattr(cls, "_FLOAT_RELU_MODULE")
            relu = cls._FLOAT_RELU_MODULE()  # type: ignore[attr-defined]
            modules.append(relu)
            fused = cls._FLOAT_MODULE(*modules)  # type: ignore[arg-type, attr-defined, operator]
            fused.train(self.training)
            return fused
        else:
            return conv


class QATDynamicConv2d(_ConvNd, nnqat.Conv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight_fake_quant: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Conv2d
    _FLOAT_CONV_MODULE = nn.Conv2d

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 qconfig=None,
                 device=None,
                 dtype=None) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            qconfig=qconfig,
            device=device,
            dtype=dtype)
        self.active_in_channels = in_channels
        self.active_out_channels = out_channels
        self.gpd_ratio = 1
        self.active_ratio = 1

    @property  # important
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        weight = self.weight[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        bias = self.bias[: self.active_out_channels].contiguous()
        bias = bias * self.active_ratio
        return bias

    @property  # important
    def active_groups(self):
        return max(self.groups // self.active_ratio, 1)

    def forward(self, input):
        self.active_in_channels = input.shape[1]
        # return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight_fake_quant(self.active_weight), self.active_bias, self.stride,
                            _pair(0), self.dilation, self.active_groups)
        return F.conv2d(input, self.weight_fake_quant(self.active_weight), self.active_bias, self.stride,
                        self.padding, self.dilation, self.active_groups)
        # return F.conv2d(input, self.active_weight, self.active_bias, self.stride,
        #                 self.padding, self.dilation, self.groups)

    @classmethod
    def from_float(cls, mod):
        return super().from_float(cls, mod)

    def export(self) -> nn.Module:  # important
        module = nnqat.Conv2d(
            self.active_in_channels,
            self.active_out_channels,
            self.kernel_size[0],
            stride=self.stride,
            # padding=get_same_padding(self.kernel_size[0]) * self.dilation[0],
            padding=self.padding,
            dilation=self.dilation,
            groups=self.active_groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            qconfig=self.qconfig,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    # ['weight', 'bias', 'weight_fake_quant.scale', 'weight_fake_quant.zero_point',
    #  'weight_fake_quant.fake_quant_enabled', 'weight_fake_quant.observer_enabled', 'weight_fake_quant.eps',
    #  'weight_fake_quant.activation_post_process.eps', 'weight_fake_quant.activation_post_process.min_val',
    #  'weight_fake_quant.activation_post_process.max_val']
    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().state_dict()
        # print(list(state_dict.keys()))
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        if "weight_fake_quant.scale" in state_dict:
            state_dict["weight_fake_quant.scale"] = state_dict["weight_fake_quant.scale"][:self.active_out_channels] * self.gpd_ratio
        if "weight_fake_quant.zero_point" in state_dict:
            state_dict["weight_fake_quant.zero_point"] = state_dict["weight_fake_quant.zero_point"][
                                                         :self.active_out_channels]
        return state_dict


class QATDynamicLinear(nnqat.Linear):
    r"""
    A linear module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
    for documentation.

    Similar to `torch.nn.Linear`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Linear
    def __init__(self, in_features, out_features, bias=True,
                 qconfig=None, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, qconfig, device, dtype)
        self.active_in_features = in_features
        self.active_out_features = out_features
        self.gpd_ratio = 1
        self.active_ratio = 1

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        return self.weight[
            : self.active_out_features, : self.active_in_features
        ].contiguous() * self.active_ratio

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias[: self.active_out_features].contiguous() * self.active_ratio

    def forward(self, input):
        self.active_in_features = input.shape[-1]
        # print(self.active_weight.shape, self.active_out_features, self.active_ratio)
        return F.linear(input, self.weight_fake_quant(self.active_weight), self.active_bias)

    def export(self) -> nn.Module:  # important
        module = nnqat.Linear(
            self.active_in_features,
            self.active_out_features,
            bias=self.bias is not None,
            qconfig=self.qconfig,
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        if "weight_fake_quant.scale" in state_dict:
            state_dict["weight_fake_quant.scale"] = state_dict["weight_fake_quant.scale"][:self.active_out_features] * self.gpd_ratio
        if "weight_fake_quant.zero_point" in state_dict:
            state_dict["weight_fake_quant.zero_point"] = state_dict["weight_fake_quant.zero_point"][
                                                         :self.active_out_features]
        return state_dict


# class QATDynamicConvTranspose2d(qnn.qat.ConvTranspose2d):
#     _FLOAT_MODULE = nn.ConvTranspose2d
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, padding=0, output_padding=0,
#                  groups=1, bias=True, dilation=1,
#                  padding_mode='zeros', qconfig=None, gpd_ratio=1):
#
#         super().__init__(in_channels, out_channels, kernel_size,
#                          stride=stride, padding=padding, output_padding=output_padding,
#                          groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode, qconfig=qconfig)
#         assert qconfig, 'qconfig must be provided for QAT module'
#         self.qconfig = qconfig
#         self.weight_fake_quant = qconfig.weight()
#         # ConvTranspose do per-channel quantize on output channel.
#         if self.weight_fake_quant.ch_axis != -1:
#             self.weight_fake_quant.ch_axis = 1
#             self.weight_fake_quant.activation_post_process.ch_axis = 1
#
#         self.active_in_channels = in_channels
#         self.active_out_channels = out_channels
#         self.gpd_ratio = gpd_ratio
#         self.active_ratio = 1
#
#     @property  # important
#     def active_weight(self) -> Optional[torch.Tensor]:
#         if self.weight is None:
#             return None
#         weight = self.weight[: self.active_in_channels, : self.active_out_channels]
#         weight = weight.contiguous()
#         if self.groups > 1:
#             weight = weight * self.gpd_ratio
#         else:
#             weight = weight * self.active_ratio
#         return weight
#
#     @property  # important
#     def active_bias(self) -> Optional[torch.Tensor]:
#         if self.bias is None:
#             return None
#         return self.bias[: self.active_out_channels].contiguous() * self.active_ratio
#
#     @property  # important
#     def active_groups(self):
#         return max(self.groups // self.active_ratio, 1)
#
#     def forward(self, x, output_size=None):
#         self.active_in_channels = x.shape[1]
#         if self.padding_mode != "zeros":
#             raise NotImplementedError
#         else:
#             pass
#
#         output_padding = self._output_padding(
#             x, output_size, self.stride, self.padding, self.kernel_size, self.dilation
#         )
#         return F.conv_transpose2d(
#             x, self.weight_fake_quant(self.active_weight), self.active_bias, self.stride, self.padding,
#             output_padding, self.active_groups, self.dilation)
#
#     def export(self) -> nn.Module:  # important
#         module = qnn.qat.ConvTranspose2d(
#             self.active_in_channels,
#             self.active_out_channels,
#             self.kernel_size[0],
#             stride=self.stride,
#             padding=self.padding,
#             output_padding=self.output_padding,
#             groups=self.active_groups,
#             bias=self.bias is not None,
#             dilation=self.dilation,
#             padding_mode=self.padding_mode,
#             qconfig=self.qconfig,
#         )
#         module.load_state_dict(self.active_state_dict())
#         return module
#
#     def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
#         state_dict = super().state_dict()
#         if self.weight is not None:
#             state_dict["weight"] = self.active_weight
#         if self.bias is not None:
#             state_dict["bias"] = self.active_bias
#         if "weight_fake_quant.scale" in state_dict:
#             state_dict["weight_fake_quant.scale"] = state_dict["weight_fake_quant.scale"][:self.active_out_channels] * self.gpd_ratio
#         if "weight_fake_quant.zero_point" in state_dict:
#             state_dict["weight_fake_quant.zero_point"] = state_dict["weight_fake_quant.zero_point"][:self.active_out_channels]
#         return state_dict


class DotProduct(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, q, k):
        attn = (q @ k.transpose(-2,-1))
        return attn


class DynamicDotProduct(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.active_ratio = 1
        self.gpd_ratio = 1

    def forward(self, q, k):
        try:
            attn = (q * self.gpd_ratio / self.active_ratio @ k.transpose(-2,-1) * self.gpd_ratio / self.active_ratio)  # * self.scale
        except:
            assert False, f"Consider putting {self.module_name} into exclude_module_name_list"
        return attn



class DynamicOREPA(DynamicModule, OREPA):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False, weight_only=False, single_branch_preserve=False, init_hyper_para=1.0, init_hyper_gamma=1.0, num_branches_orepa=6, gpd_ratio=1, in_channels_expanded=False, out_channels_expanded=False):
        OREPA.__init__(
            self,
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, internal_channels_1x1_3x3, deploy, nonlinear, single_init, weight_only, single_branch_preserve, init_hyper_para, init_hyper_gamma, num_branches_orepa
        )
        self.active_in_channels = in_channels  # important
        self.active_out_channels = out_channels  # important
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important
        self.in_channels_expanded = in_channels_expanded
        self.out_channels_expanded = out_channels_expanded
        # self.min_in_channels = in_channels
        # self.register_full_backward_hook(self._backward_grad_scale_hook)
        # param scale
        param_list = ['weight_orepa_origin', 'bias_orepa_origin', 'weight_orepa_avg_conv', 'weight_orepa_pfir_conv',
         'weight_orepa_1x1', 'weight_orepa_1x1_kxk_idconv1', 'weight_orepa_1x1_kxk_conv2', 'weight_orepa_gconv_dw',
         'weight_orepa_gconv_pw', 'vector', 'weight_orepa_avg_avg', 'id_tensor', 'weight_orepa_prior']
        self.param_active_scale = {n: 1 for n in param_list}

    def reset_param_active_scale(self):
        for k, v in self.param_active_scale.items():
            self.param_active_scale[k] = 1

    def _backward_grad_scale_hook(self, module, grad_input, grad_output):
        if self.gpd_ratio > 1 and self.active_ratio == self.gpd_ratio:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    expanded_params = getattr(self, name)
                    if expanded_params is not None:
                        expanded_params.grad = expanded_params.grad / self.param_active_scale[name]

    @property  # important
    def active_weight_orepa_origin(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_origin is None:
            return None
        weight = self.weight_orepa_origin[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_origin, self.active_out_channels, self.active_in_channels)

        if self.groups == 1:
            weight = weight * self.active_ratio
            self.param_active_scale['weight_orepa_origin'] = self.active_ratio

        return weight

    @property  # important
    def active_weight_orepa_avg_conv(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_avg_conv is None:
            return None
        weight = self.weight_orepa_avg_conv[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_avg_conv, self.active_out_channels,
        #                                     self.active_in_channels)
        if self.groups == 1:
            weight = weight * self.active_ratio
            self.param_active_scale['weight_orepa_avg_conv'] = self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_pfir_conv(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_pfir_conv is None:
            return None
        weight = self.weight_orepa_pfir_conv[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_pfir_conv, self.active_out_channels,
        #                                     self.active_in_channels)
        if self.groups == 1:
            weight = weight * self.active_ratio
            self.param_active_scale['weight_orepa_pfir_conv'] = self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_prior(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_prior is None:
            return None
        weight = self.weight_orepa_prior[: self.active_out_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model1d(weight, self.weight_orepa_prior, self.active_out_channels)
        weight = weight
        return weight

    @property  # important
    def active_weight_orepa_1x1_kxk_idconv1(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_1x1_kxk_idconv1 is None:
            return None
        weight = self.weight_orepa_1x1_kxk_idconv1[: self.active_in_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_1x1_kxk_idconv1, self.active_in_channels,
        #                                     self.active_in_channels)
        if self.out_channels_expanded:
            active_ratio = self.active_ratio if self.in_channels_expanded else 1
        else:
            active_ratio = self.in_channels // self.active_in_channels
        if self.groups == 1:
            weight = weight * active_ratio
            # self.param_active_scale['weight_orepa_1x1_kxk_idconv1'] = self.active_ratio
        return weight

    @property  # important
    def active_id_tensor(self) -> Optional[torch.Tensor]:
        if self.id_tensor is None:
            return None
        weight = self.id_tensor[: self.active_in_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.id_tensor, self.active_in_channels,
        #                                     self.active_in_channels)
        if self.out_channels_expanded:
            active_ratio = self.active_ratio if self.in_channels_expanded else 1
        else:
            active_ratio = self.in_channels // self.active_in_channels
        if self.groups == 1:
            weight = weight * active_ratio
            self.param_active_scale['id_tensor'] = self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_1x1_kxk_conv2(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_1x1_kxk_conv2 is None:
            return None
        weight = self.weight_orepa_1x1_kxk_conv2[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_1x1_kxk_conv2, self.active_out_channels,
        #                                     self.active_in_channels)
        if self.groups == 1:
            weight = weight * self.active_ratio
            self.param_active_scale['weight_orepa_1x1_kxk_conv2'] = self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_gconv_dw(self) -> Optional[torch.Tensor]:
        expand_ratio = 8
        if self.weight_orepa_gconv_dw is None:
            return None
        weight = self.weight_orepa_gconv_dw[: self.active_in_channels * expand_ratio]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model1d(weight, self.weight_orepa_gconv_dw, self.active_in_channels * expand_ratio)
        return weight

    @property  # important
    def active_weight_orepa_gconv_pw(self) -> Optional[torch.Tensor]:
        expand_ratio = 8
        if self.weight_orepa_gconv_pw is None:
            return None
        group_ratio = 1 if self.groups == 1 else self.active_ratio
        weight = self.weight_orepa_gconv_pw[: self.active_out_channels, : self.active_in_channels * expand_ratio // group_ratio]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_gconv_pw, self.active_out_channels,
        #                                     self.active_in_channels * expand_ratio // group_ratio)
        if self.groups == 1:
            weight = weight * self.active_ratio
            self.param_active_scale['weight_orepa_gconv_pw'] = self.active_ratio
        return weight

    @property  # important
    def active_vector(self) -> Optional[torch.Tensor]:
        if self.vector is None:
            return None
        weight = self.vector[:, : self.active_out_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model1d(weight, self.vector, self.active_out_channels)
        return weight

    @property  # important
    def active_weight_orepa_1x1(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_1x1 is None:
            return None
        weight = self.weight_orepa_1x1[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        # weight = avg_param_for_tiny_model2d(weight, self.weight_orepa_1x1, self.active_out_channels, self.active_in_channels)
        if self.groups == 1:
            weight = weight * self.active_ratio
            self.param_active_scale['weight_orepa_1x1'] = self.active_ratio
        return weight

    @property  # important
    def active_bias_orepa_origin(self) -> Optional[torch.Tensor]:
        if self.bias_orepa_origin is None:
            return None
        bias = self.bias_orepa_origin[: self.active_out_channels].contiguous()
        # bias = avg_param_for_tiny_model1d(bias, self.bias_orepa_origin, self.active_out_channels)
        if self.groups == 1:
            bias = bias * self.active_ratio
            self.param_active_scale['bias_orepa_origin'] = self.active_ratio
        return bias

    def active_grad_mask(self, attr_name):
        # record the current state
        original_in_channels = self.active_in_channels
        original_active_ratio = self.active_ratio
        # change state temporally
        out_attr_name = get_outchannels(self)
        full_out_channels = getattr(self, out_attr_name)
        setattr(self, 'active_' + out_attr_name, full_out_channels // self.gpd_ratio)
        self.active_ratio = self.gpd_ratio
        self.active_in_channels = self.min_in_channels
        # build grad mask
        tiny_params = getattr(self, "active_{}".format(attr_name))
        expanded_params = getattr(self, attr_name)
        grad_mask = expanded_params.new_empty(expanded_params.shape).fill_(1.)
        if len(tiny_params.shape) == 4:
            a, b, c, d = tiny_params.shape
            grad_mask[:a, :b, :c, :d].fill_(0)
        elif len(tiny_params.shape) == 3:
            a, b, c = tiny_params.shape
            grad_mask[:a, :b, :c].fill_(0)
        elif len(tiny_params.shape) == 2:
            a, b = tiny_params.shape
            grad_mask[:a, :b].fill_(0)
        elif len(tiny_params.shape) == 1:
            a = tiny_params.shape[0]
            grad_mask[:a].fill_(0)
        # reset state
        setattr(self, 'active_' + out_attr_name, full_out_channels)
        self.active_in_channels = original_in_channels
        self.active_ratio = original_active_ratio
        return grad_mask

    def apply_grad_mask(self, attr_name):
        expanded_params = getattr(self, attr_name)
        if expanded_params is not None:
            grad_mask = self.active_grad_mask(attr_name)
            expanded_params.grad = expanded_params.grad * grad_mask

    def _backward_hook(self, module, grad_input, grad_output):
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.mean().item())
        if self.active_ratio == 1:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.apply_grad_mask(name)

    @property  # important
    def active_groups(self):
        return max(self.groups // self.active_ratio, 1)

    def active_weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                           self.active_weight_orepa_origin,
                                           self.active_vector[0, :])

        weight_orepa_avg = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,hw->oihw', self.active_weight_orepa_avg_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_avg_avg), self.active_vector[1, :])

        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.active_weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                         self.active_weight_orepa_prior), self.active_vector[2, :])


        weight_orepa_1x1_kxk_conv1 = (self.active_weight_orepa_1x1_kxk_idconv1 +
                                      self.active_id_tensor).squeeze(3).squeeze(2)
        weight_orepa_1x1_kxk_conv2 = self.active_weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.active_groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(
                g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(
                g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw',
                                                weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2).reshape(
                o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw',
                                                weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.active_vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.active_weight_orepa_1x1,
                                                  self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                            self.active_vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.active_weight_orepa_gconv_dw,
                                            self.active_weight_orepa_gconv_pw,
                                            self.active_in_channels, self.active_groups)

        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                          self.active_vector[5, :])

        # weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        weights_list = [weight_orepa_origin, weight_orepa_avg, weight_orepa_pfir, weight_orepa_1x1_kxk, weight_orepa_1x1, weight_orepa_gconv]

        weight = 0
        for i in range(self.num_branches_orepa):
            weight = weight + weights_list[i % 6]

        bias = self.active_bias_orepa_origin

        return weight, bias

    def forward(self, inputs=None):
        self.reset_param_active_scale()
        self.active_in_channels = inputs.shape[1]
        # self.min_in_channels = min(self.min_in_channels, self.active_in_channels)

        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))

        weight, bias = self.active_weight_gen()
        if self.groups == 1 and bias is not None:
            bias = bias * self.gpd_ratio / self.active_ratio

        if self.weight_only is True:
            return weight

        # if self.groups > 1:
        inputs = inputs * self.gpd_ratio / self.active_ratio

        out = F.conv2d(
            inputs,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.active_groups)

        # if self.groups > 1:
        out = out / self.gpd_ratio * self.active_ratio

        return self.nonlinear(out)
        # except:
        #     assert False, f"Consider putting {self.module_name} into exclude_module_name_list"

    def export(self) -> nn.Module:  # important
        module = OREPA(
            self.active_in_channels, self.active_out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.active_groups, self.internal_channels_1x1_3x3, self.deploy, self.nonlinear, self.single_init, self.weight_only, self.single_branch_preserve, self.init_hyper_para, self.init_hyper_gamma, self.num_branches_orepa
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        param_name_list = ['weight_orepa_origin', 'bias_orepa_origin', 'weight_orepa_avg_conv', 'weight_orepa_pfir_conv', 'weight_orepa_1x1', 'weight_orepa_1x1_kxk_idconv1', 'weight_orepa_1x1_kxk_conv2', 'weight_orepa_gconv_dw', 'weight_orepa_gconv_pw', 'vector', 'weight_orepa_avg_avg', 'id_tensor', 'weight_orepa_prior']

        for param_name in param_name_list:
            try:
                state_dict[param_name] = getattr(self, 'active_' + param_name)
            except:
                state_dict[param_name] = getattr(self, param_name)

        if self.active_bias_orepa_origin is not None:
            if self.groups > 1:
                scale_factor = self.gpd_ratio / self.active_ratio
                state_dict["bias_orepa_origin"] = self.active_bias_orepa_origin / scale_factor


        return state_dict



class DynamicOREPA_SE_4x(DynamicModule, OREPA_SE_4x):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False, weight_only=False, single_branch_preserve=False, init_hyper_para=1.0, init_hyper_gamma=1.0, gpd_ratio=1, in_channels_expanded=False, out_channels_expanded=False):
        OREPA_SE_4x.__init__(
            self,
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, internal_channels_1x1_3x3, deploy, nonlinear, single_init, weight_only, single_branch_preserve, init_hyper_para, init_hyper_gamma
        )
        self.active_in_channels = in_channels  # important
        self.active_out_channels = out_channels  # important
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important
        self.in_channels_expanded = in_channels_expanded
        self.out_channels_expanded = out_channels_expanded

    @property  # important
    def active_weight_orepa_origin(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_origin is None:
            return None
        weight = self.weight_orepa_origin[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_avg_conv(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_avg_conv is None:
            return None
        weight = self.weight_orepa_avg_conv[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_pfir_conv(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_pfir_conv is None:
            return None
        weight = self.weight_orepa_pfir_conv[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_prior(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_prior is None:
            return None
        weight = self.weight_orepa_prior[: self.active_out_channels]
        weight = weight.contiguous()
        weight = weight
        return weight

    @property  # important
    def active_weight_orepa_1x1_kxk_idconv1(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_1x1_kxk_idconv1 is None:
            return None
        weight = self.weight_orepa_1x1_kxk_idconv1[: self.active_in_channels, : self.active_in_channels]
        weight = weight.contiguous()
        if self.out_channels_expanded:
            active_ratio = self.active_ratio if self.in_channels_expanded else 1
        else:
            active_ratio = self.in_channels // self.active_in_channels
        weight = weight * active_ratio
        return weight

    @property  # important
    def active_id_tensor(self) -> Optional[torch.Tensor]:
        if self.id_tensor is None:
            return None
        weight = self.id_tensor[: self.active_in_channels, : self.active_in_channels]
        weight = weight.contiguous()
        if self.out_channels_expanded:
            active_ratio = self.active_ratio if self.in_channels_expanded else 1
        else:
            active_ratio = self.in_channels // self.active_in_channels
        weight = weight * active_ratio
        return weight

    @property  # important
    def active_weight_orepa_1x1_kxk_conv2(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_1x1_kxk_conv2 is None:
            return None
        weight = self.weight_orepa_1x1_kxk_conv2[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_weight_orepa_gconv_dw(self) -> Optional[torch.Tensor]:
        expand_ratio = 8
        if self.weight_orepa_gconv_dw is None:
            return None
        weight = self.weight_orepa_gconv_dw[: self.active_in_channels * expand_ratio]
        weight = weight.contiguous()
        weight = weight
        return weight

    @property  # important
    def active_weight_orepa_gconv_pw(self) -> Optional[torch.Tensor]:
        expand_ratio = 8
        if self.weight_orepa_gconv_pw is None:
            return None
        weight = self.weight_orepa_gconv_pw[: self.active_out_channels, : self.active_in_channels * expand_ratio]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_vector(self) -> Optional[torch.Tensor]:
        if self.vector is None:
            return None
        weight = self.vector[:, : self.active_out_channels]
        weight = weight.contiguous()
        weight = weight
        return weight

    @property  # important
    def active_weight_orepa_1x1(self) -> Optional[torch.Tensor]:
        if self.weight_orepa_1x1 is None:
            return None
        weight = self.weight_orepa_1x1[: self.active_out_channels, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    @property  # important
    def active_bias_orepa_origin(self) -> Optional[torch.Tensor]:
        if self.bias_orepa_origin is None:
            return None
        bias = self.bias_orepa_origin[: self.active_out_channels].contiguous()
        bias = bias * self.active_ratio
        return bias

    def active_weight_res_deep_conv1_expand(self, w):
        weight = w[: self.active_in_channels * self._EXPAND, : self.active_in_channels]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    def active_bias_res_deep_conv1_expand(self, w):
        bias = w[: self.active_in_channels * self._EXPAND].contiguous()
        bias = bias * self.active_ratio
        return bias

    def active_weight_res_deep_conv3(self, w):
        weight = w[: self.active_in_channels * self._EXPAND, : self.active_in_channels * self._EXPAND]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    def active_bias_res_deep_conv3(self, w):
        bias = w[: self.active_in_channels * self._EXPAND].contiguous()
        bias = bias * self.active_ratio
        return bias

    def active_weight_res_deep_conv1_squeeze(self, w):
        weight = w[: self.active_out_channels, : self.active_in_channels * self._EXPAND]
        weight = weight.contiguous()
        weight = weight * self.active_ratio
        return weight

    def active_bias_res_deep_conv1_squeeze(self, w):
        bias = w[: self.active_out_channels].contiguous()
        bias = bias * self.active_ratio
        return bias

    @property  # important
    def active_scale(self):
        weight = self.scale[:, : self.active_out_channels]
        weight = weight.contiguous()
        weight = weight
        return weight

    def active_weight_gen(self):
        # single expert merge conv
        # 1x1 branch
        conv_1x1_weight = self.active_weight_orepa_origin
        conv_1x1_bias = self.active_bias_orepa_origin
        covnv_pad_3x3 = torch.nn.functional.pad(conv_1x1_weight, [1, 1, 1, 1])  # 单独的conv分支，可以理解为conv2
        conv_weight = covnv_pad_3x3
        conv_bias = conv_1x1_bias

        for res_deep_conv3_stack, bias_res_deep_conv3_stack in zip(self.res_deep_conv3_stack_list, self.bias_res_deep_conv3_stack_list):
            conv_deep_weight, conv_deep_bias = merge_conv(
                self.active_weight_res_deep_conv1_expand(res_deep_conv3_stack[0]),
                self.active_bias_res_deep_conv1_expand(bias_res_deep_conv3_stack[0]),
                self.active_weight_res_deep_conv3(res_deep_conv3_stack[1]),
                self.active_bias_res_deep_conv3(bias_res_deep_conv3_stack[1]),
            )
            conv_deep_weight, conv_deep_bias = merge_conv(
                conv_deep_weight,
                conv_deep_bias,
                self.active_weight_res_deep_conv1_squeeze(res_deep_conv3_stack[2]),
                self.active_bias_res_deep_conv1_squeeze(bias_res_deep_conv3_stack[2]),
            )
            # TODO
            conv_weight = conv_weight + conv_deep_weight
            conv_bias = conv_bias + conv_deep_bias

        out_plane_conv_weight, in_plane_conv_weight = conv_weight.shape[0], conv_weight.shape[1]
        _added = torch.zeros_like(conv_weight[:out_plane_conv_weight // self.gpd_ratio * self.active_ratio, :in_plane_conv_weight // self.gpd_ratio * self.active_ratio])
        nn.init.dirac_(_added)

        if self.gpd_ratio > 1:
            if self.active_ratio == 1:
                _added = torch.cat(
                    [_added / self.gpd_ratio for i in range(self.gpd_ratio)], 0)
                _added = torch.cat(
                    [_added for i in range(self.gpd_ratio)], 1)
            else:
                _added = _added / self.gpd_ratio
        _added = _added * self.active_ratio

        conv_weight = conv_weight * self.active_scale.permute(1, 0, 2, 3) + _added
        conv_bias = conv_bias * self.active_scale.reshape(-1)
        weight_orepa_origin = conv_weight
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                           weight_orepa_origin,
                                           self.active_vector[0, :])

        weight_orepa_avg = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,hw->oihw', self.active_weight_orepa_avg_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_avg_avg), self.active_vector[1, :])

        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.active_weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                         self.active_weight_orepa_prior), self.active_vector[2, :])


        weight_orepa_1x1_kxk_conv1 = (self.active_weight_orepa_1x1_kxk_idconv1 +
                                      self.active_id_tensor).squeeze(3).squeeze(2)

        weight_orepa_1x1_kxk_conv2 = self.active_weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(
                g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(
                g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw',
                                                weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2).reshape(
                o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw',
                                                weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.active_vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.active_weight_orepa_1x1,
                                                  self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                            self.active_vector[4, :] * 0)

        weight_orepa_gconv = self.dwsc2full(self.active_weight_orepa_gconv_dw,
                                            self.active_weight_orepa_gconv_pw,
                                            self.active_in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                          self.active_vector[5, :])

        # print(weight_orepa_origin.shape, weight_orepa_avg.shape, weight_orepa_1x1.shape, weight_orepa_1x1_kxk.shape, weight_orepa_pfir.shape, weight_orepa_gconv.shape, self.active_in_channels)
        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        bias = conv_bias

        return weight, bias

    def forward(self, inputs=None):
        self.active_in_channels = inputs.shape[1]

        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))

        weight, bias = self.active_weight_gen()

        if self.weight_only is True:
            return weight

        out = F.conv2d(
            inputs,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return self.nonlinear(out)

    def export(self) -> nn.Module:  # important
        nonlinear = self.nonlinear if isinstance(self.nonlinear, nn.Identity) else self.nonlinear.export()
        module = OREPA_SE_4x(
            self.active_in_channels, self.active_out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.internal_channels_1x1_3x3, self.deploy, nonlinear, self.single_init, self.weight_only, self.single_branch_preserve, self.init_hyper_para, self.init_hyper_gamma
        )

        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        param_name_list = ['weight_orepa_origin', 'bias_orepa_origin', 'weight_orepa_avg_conv', 'weight_orepa_pfir_conv', 'weight_orepa_1x1', 'weight_orepa_1x1_kxk_idconv1', 'weight_orepa_1x1_kxk_conv2', 'weight_orepa_gconv_dw', 'weight_orepa_gconv_pw', 'vector', 'weight_orepa_avg_avg', 'id_tensor', 'weight_orepa_prior', 'scale']
        for param_name in param_name_list:
            try:
                state_dict[param_name] = getattr(self, 'active_' + param_name)
            except:
                state_dict[param_name] = getattr(self, param_name)

        new_res_deep_conv3_stack_list = nn.ModuleList()
        new_bias_res_deep_conv3_stack_list = nn.ModuleList()
        for i in range(self._3CONV_BR_NUM):
            res_deep_conv3_stack = self.res_deep_conv3_stack_list[i]
            bias_res_deep_conv3_stack = self.bias_res_deep_conv3_stack_list[i]
            # conv1
            res_deep_conv1_expand = self.active_weight_res_deep_conv1_expand(res_deep_conv3_stack[0])
            bias_res_deep_conv1_expand = self.active_bias_res_deep_conv1_expand(bias_res_deep_conv3_stack[0])
            # conv2
            res_deep_conv3 = self.active_weight_res_deep_conv3(res_deep_conv3_stack[1])
            bias_res_deep_conv3 = self.active_bias_res_deep_conv3(bias_res_deep_conv3_stack[1])
            # conv3
            res_deep_conv1_squeeze = self.active_weight_res_deep_conv1_squeeze(res_deep_conv3_stack[2])
            bias_res_deep_conv1_squeeze = self.active_bias_res_deep_conv1_squeeze(bias_res_deep_conv3_stack[2])

            new_res_deep_conv3_stack_list.append(
                nn.ParameterList([res_deep_conv1_expand, res_deep_conv3, res_deep_conv1_squeeze]))
            new_bias_res_deep_conv3_stack_list.append(
                nn.ParameterList([bias_res_deep_conv1_expand, bias_res_deep_conv3, bias_res_deep_conv1_squeeze]))

        new_res_deep_conv3_stack_list_state_dict = new_res_deep_conv3_stack_list.state_dict()
        for name in new_res_deep_conv3_stack_list_state_dict.keys():
            param_name = 'res_deep_conv3_stack_list.' + name
            state_dict[param_name] = new_res_deep_conv3_stack_list_state_dict[name]

        new_bias_res_deep_conv3_stack_list_state_dict = new_bias_res_deep_conv3_stack_list.state_dict()
        for name in new_bias_res_deep_conv3_stack_list_state_dict.keys():
            param_name = 'bias_res_deep_conv3_stack_list.' + name
            state_dict[param_name] = new_bias_res_deep_conv3_stack_list_state_dict[name]

        # print(state_dict['nonlinear.weight'].shape, self.nonlinear.active_weight.shape)
        # state_dict['nonlinear.weight'] = self.nonlinear.active_weight

        return state_dict


class DynamicTokenMixing(DynamicModule, TokenMixing):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                    sr_ratio=1, linear=False, share_atten=False, drop_path=0., emlp=False, sharpen_attn=False,
                    mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, gpd_ratio=1):
        TokenMixing.__init__(
            self, dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio, linear=linear, share_atten=share_atten, drop_path=drop_path, emlp=emlp, sharpen_attn=sharpen_attn, mlp_hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_layer=norm_layer
        )
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_dim(self):
        active_dim = self.dim // self.active_ratio
        return active_dim

    @property  # important
    def active_num_heads(self):
        active_num_heads = self.num_heads // self.active_ratio
        return active_num_heads

    def forward(self, x, H, W, atten=None, return_attention=False):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)
        if self.active_ratio > 1:
            kv = self.kv(x).reshape(B, -1, 2, self.active_num_heads, C // self.active_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            kv = self.kv(x).reshape(B, -1, 2 * self.gpd_ratio, self.active_num_heads // self.gpd_ratio, C // self.active_num_heads).permute(2, 0, 3, 1, 4)
            if kv.shape[0] > 2:
                k = torch.cat([kv[i * 2] for i in range(self.gpd_ratio)], dim=1)
                v = torch.cat([kv[i * 2 + 1] for i in range(self.gpd_ratio)], dim=1)
            else:
                k, v = kv[0], kv[1]

        q = q * self.gpd_ratio / self.active_ratio
        k = k * self.gpd_ratio / self.active_ratio

        attn = (q * self.scale @ k.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # return x, attn  @ v
        return x, None

    def export(self) -> nn.Module:  # important
        module = TokenMixing(
                dim=self.active_dim, num_heads=self.active_num_heads, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop=self.attn_drop_prob, proj_drop=self.proj_drop_prob, sr_ratio=self.sr_ratio, linear=self.linear, share_atten=self.share_atten, emlp=self.emlp)
        contain_orepa = check_contain_orepa(self)
        if contain_orepa:
            module = transfer2orepa(module, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        return state_dict

    def extra_repr(self):
        s = ('{active_dim}, {active_num_heads}, {gpd_ratio}')
        return s.format(**self.__dict__)


class DynamicAttention(DynamicModule, Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_mask=False, gpd_ratio=1):
        Attention.__init__(
            self, dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, use_mask=use_mask
        )
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_dim(self):
        active_dim = self.dim // self.active_ratio
        return active_dim

    @property  # important
    def active_num_heads(self):
        active_num_heads = self.num_heads // self.active_ratio
        return active_num_heads

    def forward(self, x):
        B, N, C = x.shape

        if self.active_ratio > 1:
            qkv = self.qkv(x).reshape(B, N, 3, self.active_num_heads, C // self.active_num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = self.qkv(x).reshape(B, N, 3 * self.gpd_ratio, self.active_num_heads // self.gpd_ratio, C // self.active_num_heads).permute(2, 0, 3, 1, 4)
            if qkv.shape[0] > 3:
                q = torch.cat([qkv[i * 3] for i in range(self.gpd_ratio)], dim=1)
                k = torch.cat([qkv[i * 3 + 1] for i in range(self.gpd_ratio)], dim=1)
                v = torch.cat([qkv[i * 3 + 2] for i in range(self.gpd_ratio)], dim=1)
            else:
                q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.gpd_ratio / self.active_ratio
        k = k * self.gpd_ratio / self.active_ratio

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask[:,:N,:N]).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def export(self) -> nn.Module:  # important
        module = Attention(
                dim=self.active_dim, num_heads=self.active_num_heads, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop=self.attn_drop_prob, proj_drop=self.proj_drop_prob, use_mask=self.use_mask)
        contain_orepa = check_contain_orepa(self)
        if contain_orepa:
            module = transfer2orepa(module, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        return state_dict

    def extra_repr(self):
        s = ('{dim}, {num_heads}, {gpd_ratio}')
        return s.format(**self.__dict__)


class DynamicLayerNorm(DynamicModule, nn.LayerNorm):

    def __init__(
        self,
        normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None,
        gpd_ratio: int = 1,
    ) -> None:
        nn.LayerNorm.__init__(
            self,
            normalized_shape, eps, elementwise_affine, device, dtype
        )
        self.active_num_features = normalized_shape
        self.gpd_ratio = gpd_ratio
        self.active_ratio = 1

    @property
    def active_normalized_shape(self):
        normalized_shape = (self.active_num_features,)  # type: ignore[assignment]
        active_normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        return active_normalized_shape

    @property
    def active_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None:
            return None
        # return self.weight[: self.active_num_features]
        weight = self.weight[: self.active_num_features]
        # weight = avg_param_for_tiny_model1d(weight, self.weight, self.active_num_features)
        return weight

    @property
    def active_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        # return self.bias[: self.active_num_features]
        bias = self.bias[: self.active_num_features]
        # bias = avg_param_for_tiny_model1d(bias, self.bias, self.active_num_features)
        return bias

    @property
    def active_eps(self):
        return self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            self.active_num_features = x.shape[-1]
        elif len(x.shape) == 3:
            self.active_num_features = x.shape[-1]
        elif len(x.shape) == 4:
            self.active_num_features = x.shape[1]
        else:
            assert False, x.shape
        # preprocess
        x = x * self.gpd_ratio / self.active_ratio
        try:
            output = F.layer_norm(
                x, self.active_normalized_shape, self.active_weight, self.active_bias, self.active_eps)
        except:
            output = F.layer_norm(
                x.permute(0, 2, 3, 1), self.active_normalized_shape, self.active_weight, self.active_bias, self.active_eps).permute(0, 3, 1, 2)
        # restore with scale
        output = output / self.gpd_ratio * self.active_ratio
        return output

    def export(self) -> nn.Module:
        module = getattr(nn, "LayerNorm")(
            self.active_num_features,
            eps=self.active_eps,
            elementwise_affine=self.elementwise_affine
        )
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        state_dict = super().active_state_dict()
        if self.weight is not None:
            state_dict["weight"] = self.active_weight
        if self.bias is not None:
            state_dict["bias"] = self.active_bias
        return state_dict


class DynamicGELU(DynamicModule, nn.GELU):
    def __init__(self, approximate: str = 'none', gpd_ratio=1) -> None:
        try:
            nn.GELU.__init__(self, approximate)
        except:
            nn.GELU.__init__(self)
        self.gpd_ratio = gpd_ratio
        self.active_ratio = 1

    def forward(self, x):
        # preprocess
        x = x * self.gpd_ratio / self.active_ratio
        try:
            output = F.gelu(x, approximate=self.approximate)
        except:
            output = F.gelu(x)
        # restore with scale
        output = output / self.gpd_ratio * self.active_ratio
        return output

    def export(self) -> nn.Module:  # important
        try:
            module = nn.GELU(self.approximate)
        except:
            module = nn.GELU()
        return module


class DynamicChannelProcessing(DynamicModule, ChannelProcessing):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., linear=False, drop_path=0.,
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1, c_head_num=None, gpd_ratio=1):
        ChannelProcessing.__init__(
            self, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, linear=linear, drop_path=drop_path,
                 mlp_hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_layer=norm_layer, cha_sr_ratio=cha_sr_ratio, c_head_num=c_head_num
        )
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_dim(self):
        active_dim = self.dim // self.active_ratio
        return active_dim

    @property  # important
    def active_num_heads(self):
        active_num_heads = self.num_heads // self.active_ratio
        return active_num_heads

    @property  # important
    def active_temperature(self):
        # active_temperature = self.temperature[:self.active_num_heads]
        active_temperature = self.temperature[: self.active_num_heads]
        # active_temperature = avg_param_for_tiny_model1d(active_temperature, self.temperature, self.active_num_heads)
        return active_temperature

    @property  # important
    def active_mlp_v_gamma(self):
        new_shape = self.mlp_v.gamma.shape[0]
        new_shape = new_shape // self.active_ratio
        # active_mlp_v_gamma = self.mlp_v.gamma[:new_shape]
        active_mlp_v_gamma = self.mlp_v.gamma[: new_shape]
        # active_mlp_v_gamma = avg_param_for_tiny_model1d(active_mlp_v_gamma, self.mlp_v.gamma, new_shape)
        return active_mlp_v_gamma

    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1, -2)
        _, _, N, _ = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))
        attn = torch.nn.functional.sigmoid(q @ k)
        return attn * self.active_temperature

    def mlp_forward(self, x, H, W):
        x = self.mlp_v.fc1(x)
        if self.mlp_v.linear:
            x = self.mlp_v.relu(x)
        x = self.mlp_v.drop(self.active_mlp_v_gamma * self.mlp_v.dwconv(x, H, W)) + x
        x = self.mlp_v.fc2(x)
        x = self.mlp_v.drop(x)
        return x

    def forward(self, x, H, W, atten=None):
        # preprocess
        x = x * self.gpd_ratio / self.active_ratio

        B, N, C = x.shape
        v = x.reshape(B, N, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)

        q = self.q(x).reshape(B, N, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)

        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)

        v = v / self.gpd_ratio * self.active_ratio
        Bv, Hd, Nv, Cv = v.shape

        v = self.mlp_forward(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv), H, W)
        v = self.norm_v(v).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x,  (attn * v.transpose(-1, -2)).transpose(-1, -2) #attn

    def export(self) -> nn.Module:  # important
        module = ChannelProcessing(dim=self.active_dim, num_heads=self.active_num_heads, qkv_bias=self.qkv_bias, attn_drop=self.attn_drop_prob, linear=self.linear, drop_path=self.drop_path_prob, mlp_hidden_dim=self.mlp_hidden_dim, act_layer=self.act_layer, drop=self.drop, norm_layer=self.norm_layer, cha_sr_ratio=self.cha_sr_ratio, c_head_num=self.c_head_num)
        contain_orepa = check_contain_orepa(self)
        if contain_orepa:
            module = transfer2orepa(module, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
        module.load_state_dict(self.active_state_dict(), strict=False)
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        state_dict["temperature"] = self.active_temperature
        state_dict["mlp_v.gamma"] = self.active_mlp_v_gamma
        return state_dict

    def extra_repr(self):
        # s = ('{dim}, {num_heads}, {gpd_ratio}')
        # return s.format(**self.__dict__)
        s = f'{self.dim}, {self.num_heads}, {self.gpd_ratio}, mlp_v_gamma={self.active_mlp_v_gamma.shape[0]}, temperature={self.active_temperature.shape[0]}'
        return s


class DynamicConvNeXtBlock(DynamicModule, ConvNeXtBlock):
    def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=True, mlp_ratio=4, norm_layer=None, gpd_ratio=1):
        ConvNeXtBlock.__init__(
            self, dim, drop_path=drop_path, ls_init_value=ls_init_value, conv_mlp=conv_mlp, mlp_ratio=mlp_ratio, norm_layer=norm_layer
        )
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_dim(self):
        active_dim = self.dim // self.active_ratio
        return active_dim

    @property  # important
    def active_gamma(self):
        new_shape = self.gamma.shape[0]
        new_shape = new_shape // self.active_ratio
        active_gamma = self.gamma[:new_shape]
        # active_gamma = avg_param_for_tiny_model1d(active_gamma, self.gamma, new_shape)
        return active_gamma

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.active_gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

    def export(self) -> nn.Module:  # important
        module = ConvNeXtBlock(dim=self.active_dim, drop_path=self.drop_path_prob, ls_init_value=self.ls_init_value, conv_mlp=self.conv_mlp, mlp_ratio=self.mlp_ratio, norm_layer=self.norm_layer)
        contain_orepa = check_contain_orepa(self)
        if contain_orepa:
            module = transfer2orepa(module, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
        module.load_state_dict(self.active_state_dict())
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        state_dict["gamma"] = self.active_gamma
        return state_dict


class DynamicFANBlock(DynamicModule, FANBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., sharpen_attn=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eta=1., sr_ratio=1., downsample=None, c_head_num=None, gpd_ratio=1):
        FANBlock.__init__(
            self, dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, sharpen_attn=sharpen_attn, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, eta=eta, sr_ratio=sr_ratio, downsample=downsample, c_head_num=c_head_num
        )
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_dim(self):
        active_dim = self.dim // self.active_ratio
        return active_dim

    @property  # important
    def active_num_heads(self):
        active_num_heads = self.num_heads // self.active_ratio
        return active_num_heads

    @property  # important
    def active_gamma1(self):
        new_shape = self.gamma1.shape[0]
        new_shape = new_shape // self.active_ratio
        active_gamma1 = self.gamma1[:new_shape]
        # active_gamma1 = avg_param_for_tiny_model1d(active_gamma1, self.gamma1, new_shape)
        return active_gamma1

    @property  # important
    def active_gamma2(self):
        new_shape = self.gamma2.shape[0]
        new_shape = new_shape // self.active_ratio
        active_gamma2 = self.gamma2[:new_shape]
        # active_gamma2 = avg_param_for_tiny_model1d(active_gamma2, self.gamma2, new_shape)
        return active_gamma2

    def forward(self, x, attn=None, return_attention=False):
        H, W = self.H, self.W

        x_new, attn_s = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(self.active_gamma1 * x_new)

        x_new, attn_c = self.mlp(self.norm2(x), H, W, atten=attn)
        x = x + self.drop_path(self.active_gamma2 * x_new)

        if return_attention:
            return x, attn_s

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        self.H, self.W = H, W
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

    def export(self) -> nn.Module:  # important
        module = FANBlock(self.active_dim, self.active_num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop, sharpen_attn=self.sharpen_attn, drop_path=self.drop_path_prob, act_layer=self.act_layer, norm_layer=self.norm_layer, eta=self.eta, sr_ratio=self.sr_ratio, downsample=self.downsample, c_head_num=self.c_head_num)
        contain_orepa = check_contain_orepa(self)
        if contain_orepa:
            module = transfer2orepa(module, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
        module.H = self.H
        module.W = self.W
        module.load_state_dict(self.active_state_dict(), strict=False)
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        state_dict["gamma1"] = self.active_gamma1
        state_dict["gamma2"] = self.active_gamma2
        return state_dict


class DynamicClassAttn(DynamicModule, ClassAttn):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., gpd_ratio=1):
        ClassAttn.__init__(
            self, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.gpd_ratio = gpd_ratio  # important
        self.active_ratio = 1  # important

    @property  # important
    def active_dim(self):
        active_dim = self.dim // self.active_ratio
        return active_dim

    @property  # important
    def active_num_heads(self):
        active_num_heads = self.num_heads // self.active_ratio
        return active_num_heads

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)

        q = q * self.scale

        q = q * self.gpd_ratio / self.active_ratio
        k = k * self.gpd_ratio / self.active_ratio

        v = self.v(x).reshape(B, N, self.active_num_heads, C // self.active_num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls

    def export(self) -> nn.Module:  # important
        module = ClassAttn(self.active_dim, self.active_num_heads, qkv_bias=self.qkv_bias, attn_drop=self.attn_drop_prob, proj_drop=self.proj_drop_prob)
        contain_orepa = check_contain_orepa(self)
        if contain_orepa:
            module = transfer2orepa(module, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
        module.load_state_dict(self.active_state_dict(), strict=False)
        return module

    def active_state_dict(self) -> OrderedDict[str, torch.Tensor]:  # important
        state_dict = super().active_state_dict()
        return state_dict


def avg_param_for_tiny_model2d(weight, original_param, active_out_channels, active_in_channels):
    original_shape = original_param.shape
    active_shape = weight.shape
    out_replica_num = original_shape[0] // active_shape[0]
    in_replica_num = original_shape[1] // active_shape[1]
    if out_replica_num > 1 and in_replica_num > 1:
        assert out_replica_num == in_replica_num, f'in_replica_num is not equal to out_replica_num'
        weight = 0
        for i in range(out_replica_num):
            for j in range(in_replica_num):
                weight += original_param[i * active_out_channels: (i + 1) * active_out_channels,
                      j * active_in_channels: (j + 1) * active_in_channels]
        weight = weight / (out_replica_num * in_replica_num)
        # weight = sum([original_param[i * active_out_channels: (i + 1) * active_out_channels,
        #               i * active_in_channels: (i + 1) * active_in_channels] for i in
        #               range(out_replica_num)]) / out_replica_num
    if out_replica_num > 1 and in_replica_num == 1:
        weight = sum([original_param[i * active_out_channels: (i + 1) * active_out_channels,
                      : active_in_channels] for i in range(out_replica_num)]) / out_replica_num
    if out_replica_num == 1 and in_replica_num > 1:
        weight = sum([original_param[: active_out_channels,
                      i * active_in_channels: (i + 1) * active_in_channels] for i in
                      range(in_replica_num)]) / in_replica_num
    return weight


def avg_param_for_tiny_model1d(weight, original_param, active_out_channels):
    original_shape = original_param.shape
    active_shape = weight.shape
    out_replica_num = original_shape[0] // active_shape[0]
    if out_replica_num > 1:
        weight = sum([original_param[i * active_out_channels: (i + 1) * active_out_channels] for i in
                      range(out_replica_num)]) / out_replica_num
    return weight

