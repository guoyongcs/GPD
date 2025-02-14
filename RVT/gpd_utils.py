import copy
import torch
import torch.nn as nn
from torch.nn import Parameter
from gpd_layers import DynamicConv2d, DynamicLinear, DynamicConvTranspose2d, DynamicBatchNorm2d, DynamicPReLU, get_inchannels, get_outchannels, DotProduct, DynamicDotProduct, scale_module, DynamicScale, DynamicOREPA, DynamicOREPA_SE_4x, DynamicTokenMixing, DynamicLayerNorm, get_attr_value, DynamicGELU, DynamicChannelProcessing, DynamicConvNeXtBlock, DynamicFANBlock, DynamicClassAttn, check_contain_orepa, avg_param_for_tiny_model1d, DynamicAttention
import warnings
import torch.nn.functional as F
from orepa_ft import OREPA, OREPA_SE_4x, transfer2orepa
import numpy as np
from scipy.optimize import minimize_scalar
from models.fan import TokenMixing, ChannelProcessing, FANBlock, FAN, ClassAttn, ClassAttentionBlock
from models.convnext_utils import ConvNeXtBlock
from robust_models import Attention


static_type = (
    nn.Conv2d,
    nn.Linear,
    nn.ConvTranspose2d,
    nn.BatchNorm2d,
    nn.PReLU,
    DotProduct,
    scale_module,
    OREPA,
    OREPA_SE_4x,
    TokenMixing,
    nn.LayerNorm,
    nn.GELU,
    ChannelProcessing,
    ConvNeXtBlock,
    FANBlock,
    ClassAttn,
    Attention,
)

dynamic_type = (
    DynamicConv2d,
    DynamicLinear,
    DynamicConvTranspose2d,
    DynamicBatchNorm2d,
    DynamicPReLU,
    DynamicDotProduct,
    DynamicScale,
    DynamicOREPA,
    DynamicOREPA_SE_4x,
    DynamicTokenMixing,
    DynamicLayerNorm,
    DynamicGELU,
    DynamicChannelProcessing,
    DynamicConvNeXtBlock,
    DynamicFANBlock,
    DynamicClassAttn,
    DynamicAttention,
)


noise_scale = 1

# quant_dynamic_type = (
#     QATDynamicConv2d,
#     QATDynamicLinear,
#     QATDynamicConvTranspose2d,
# )


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
        if hasattr(m, 'groups') and m.groups > 1:
            new_weight = torch.cat([new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
        else:
            new_weight = torch.cat([new_weight / expand_outplane_ratio + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
    if expand_inplane_ratio > 1:
        new_weight = torch.cat([new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        assert expand_outplane_ratio * old_bias.shape[0] == new_bias.shape[
            0], f'old_bias {expand_outplane_ratio * old_bias.shape[0]} has different shape with new_bias {new_bias.shape[0]}'
        new_bias = m.bias.data
        if expand_outplane_ratio > 1:
            if hasattr(m, 'groups') and m.groups > 1:
                new_bias = torch.cat([new_bias + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in range(expand_outplane_ratio)], 0)
            else:
                new_bias = torch.cat([new_bias / expand_outplane_ratio + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in range(expand_outplane_ratio)], 0)
        new_m.bias.data.copy_(new_bias)


def init_expand_deconv_module(m, new_m):
    # old config
    old_inplane, old_outplane = m.weight.shape[0], m.weight.shape[1]
    old_bias = m.bias
    # new config
    new_inplane, new_outplane = new_m.weight.shape[0], new_m.weight.shape[1]
    new_bias = new_m.bias
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight / expand_outplane_ratio + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 1)
    if expand_inplane_ratio > 1:
        new_weight = torch.cat([new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 0)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        assert expand_outplane_ratio * old_bias.shape[0] == new_bias.shape[
            0], f'old_bias {expand_outplane_ratio * old_bias.shape[0]} has different shape with new_bias {new_bias.shape[0]}'
        new_bias = m.bias.data
        if expand_outplane_ratio > 1:
            new_bias = torch.cat([new_bias / expand_outplane_ratio + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in range(expand_outplane_ratio)], 0)
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
        new_weight = torch.cat([new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    new_bias = m.bias.data
    if expand_outplane_ratio > 1:
        new_bias = torch.cat([new_bias + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in range(expand_outplane_ratio)], 0)
    new_m.bias.data.copy_(new_bias)
    # init new running_mean
    new_running_mean = m.running_mean.data
    if expand_outplane_ratio > 1:
        new_running_mean = torch.cat([new_running_mean for i in range(expand_outplane_ratio)],
                                     0)
    new_m.running_mean.data.copy_(new_running_mean)
    # init new running_mean
    new_running_var = m.running_var.data
    if expand_outplane_ratio > 1:
        new_running_var = torch.cat(
            [new_running_var for i in range(expand_outplane_ratio)], 0)
    new_m.running_var.data.copy_(new_running_var)
    new_m.eps = m.eps


def init_expand_ln(m, new_m):
    # old config
    old_outplane = m.weight.shape[0]
    # new config
    new_outplane = new_m.weight.shape[0]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    new_bias = m.bias.data
    if expand_outplane_ratio > 1:
        new_bias = torch.cat([new_bias + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in range(expand_outplane_ratio)], 0)
    new_m.bias.data.copy_(new_bias)
    new_m.eps = m.eps


def init_expand_prelu(m, new_m):
    # old config
    old_outplane = m.weight.shape[0]
    # new config
    new_outplane = new_m.weight.shape[0]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
    new_m.weight.data.copy_(new_weight)


def init_expand_scale(m, new_m):
    new_shape = m.weight.shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = m.weight.shape[valid_dim]
    # new config
    new_outplane = new_m.weight.shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    new_m.weight.data.copy_(new_weight)


def init_expand_channelprocessing(m, new_m):
    # temperature
    new_shape = m.temperature.shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = m.temperature.shape[valid_dim]
    # new config
    new_outplane = new_m.temperature.shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.temperature.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    new_m.temperature.data.copy_(new_weight)
    # mlp_v.gamma
    new_shape = m.mlp_v.gamma.shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = m.mlp_v.gamma.shape[valid_dim]
    # new config
    new_outplane = new_m.mlp_v.gamma.shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.mlp_v.gamma.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    new_m.mlp_v.gamma.data.copy_(new_weight)


def init_expand_convnextblock(m, new_m):
    # gamma
    new_shape = m.gamma.shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = m.gamma.shape[valid_dim]
    # new config
    new_outplane = new_m.gamma.shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.gamma.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    new_m.gamma.data.copy_(new_weight)


def init_expand_fanblock(m, new_m):
    # gamma1
    new_shape = m.gamma1.shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = m.gamma1.shape[valid_dim]
    # new config
    new_outplane = new_m.gamma1.shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.gamma1.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    new_m.gamma1.data.copy_(new_weight)
    # gamma2
    new_shape = m.gamma2.shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = m.gamma2.shape[valid_dim]
    # new config
    new_outplane = new_m.gamma2.shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = m.gamma2.data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    new_m.gamma2.data.copy_(new_weight)

def expand_conv_weight_with_param(m, new_m, groups=1):
    old_outplane, old_inplane = m.shape[0], m.shape[1]
    new_outplane, new_inplane = new_m.shape[0], new_m.shape[1]
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = m.data
    if expand_outplane_ratio > 1:
        # TODO
        if groups > 1:
            new_weight = torch.cat(
                [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
        else:
            new_weight = torch.cat(
                [new_weight / expand_outplane_ratio + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_outplane_ratio)], 0)
    if expand_inplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    new_m.data.copy_(new_weight)


def expand_conv_bias_with_param(m, new_m, groups=1):
    old_outplane = m.shape[0]
    new_outplane = new_m.shape[0]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_bias = m.data
    if expand_outplane_ratio > 1:
        if groups > 1:
            new_bias = torch.cat(
                [new_bias + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in
                 range(expand_outplane_ratio)], 0)
        else:
            new_bias = torch.cat(
                [new_bias / expand_outplane_ratio + noise_scale * torch.normal(0, 1e-6, size=new_bias.shape) for i in
                 range(expand_outplane_ratio)], 0)
    new_m.data.copy_(new_bias)

def init_expand_orepa(m, new_m):
    conv_param_list = ['weight_orepa_origin', 'weight_orepa_avg_conv', 'weight_orepa_pfir_conv', 'weight_orepa_1x1_kxk_conv2', 'weight_orepa_gconv_pw', 'weight_orepa_1x1', 'weight_orepa_1x1_kxk_idconv1', 'id_tensor']

    groups = 1
    for conv_param in conv_param_list:
        groups = m.groups
        expand_conv_weight_with_param(getattr(m, conv_param), getattr(new_m, conv_param), groups)

    # weight_orepa_prior
    old_outplane = getattr(m, 'weight_orepa_prior').shape[0]
    new_outplane = getattr(new_m, 'weight_orepa_prior').shape[0]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = getattr(m, 'weight_orepa_prior').data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in
             range(expand_outplane_ratio)], 0)
    getattr(new_m, 'weight_orepa_prior').data.copy_(new_weight)

    # weight_orepa_gconv_dw
    old_outplane, old_inplane = getattr(m, 'weight_orepa_gconv_dw').shape[0], getattr(m, 'weight_orepa_gconv_dw').shape[1]
    new_outplane, new_inplane = getattr(new_m, 'weight_orepa_gconv_dw').shape[0], getattr(new_m, 'weight_orepa_gconv_dw').shape[1]
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = getattr(m, 'weight_orepa_gconv_dw').data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in
             range(expand_outplane_ratio)], 0)
    if expand_inplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    getattr(new_m, 'weight_orepa_gconv_dw').data.copy_(new_weight)

    # vector
    old_inplane = getattr(m, 'vector').shape[1]
    new_inplane = getattr(new_m, 'vector').shape[1]
    expand_inplane_ratio = new_inplane // old_inplane
    # init new weight
    new_weight = getattr(m, 'vector').data
    if expand_inplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    getattr(new_m, 'vector').data.copy_(new_weight)

    # bias_orepa_origin
    expand_conv_bias_with_param(getattr(m, 'bias_orepa_origin'), getattr(new_m, 'bias_orepa_origin'), groups)

def init_expand_orepa_se_4x(m, new_m):
    conv_param_with_group_list = ['weight_orepa_origin', 'weight_orepa_avg_conv', 'weight_orepa_pfir_conv', 'weight_orepa_1x1', 'weight_orepa_1x1_kxk_idconv1', 'id_tensor', 'weight_orepa_1x1_kxk_conv2', 'weight_orepa_gconv_pw']

    conv_param_list = ['weight_orepa_origin', 'weight_orepa_avg_conv', 'weight_orepa_pfir_conv', 'weight_orepa_1x1_kxk_conv2', 'weight_orepa_gconv_pw', 'weight_orepa_1x1', 'weight_orepa_1x1_kxk_idconv1', 'id_tensor', 'weight_orepa_gconv_pw']

    for conv_param in conv_param_list:
        if conv_param in conv_param_with_group_list:
            groups = m.groups
        else:
            groups = 1
        expand_conv_weight_with_param(getattr(m, conv_param), getattr(new_m, conv_param), groups)

    # weight_orepa_prior
    old_outplane = getattr(m, 'weight_orepa_prior').shape[0]
    new_outplane = getattr(new_m, 'weight_orepa_prior').shape[0]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = getattr(m, 'weight_orepa_prior').data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in
             range(expand_outplane_ratio)], 0)
    getattr(new_m, 'weight_orepa_prior').data.copy_(new_weight)

    # weight_orepa_gconv_dw
    old_outplane, old_inplane = getattr(m, 'weight_orepa_gconv_dw').shape[0], \
    getattr(m, 'weight_orepa_gconv_dw').shape[1]
    new_outplane, new_inplane = getattr(new_m, 'weight_orepa_gconv_dw').shape[0], \
    getattr(new_m, 'weight_orepa_gconv_dw').shape[1]
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = getattr(m, 'weight_orepa_gconv_dw').data
    if expand_outplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in
             range(expand_outplane_ratio)], 0)
    if expand_inplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    getattr(new_m, 'weight_orepa_gconv_dw').data.copy_(new_weight)

    # vector
    old_inplane = getattr(m, 'vector').shape[1]
    new_inplane = getattr(new_m, 'vector').shape[1]
    expand_inplane_ratio = new_inplane // old_inplane
    # init new weight
    new_weight = getattr(m, 'vector').data
    if expand_inplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    getattr(new_m, 'vector').data.copy_(new_weight)

    # bias_orepa_origin
    expand_conv_bias_with_param(getattr(m, 'bias_orepa_origin'), getattr(new_m, 'bias_orepa_origin'))

    # _3CONV_BR_NUM
    for i in range(m._3CONV_BR_NUM):
        # res_deep_conv1_expand
        expand_conv_weight_with_param(m.res_deep_conv3_stack_list[i][0], new_m.res_deep_conv3_stack_list[i][0])
        # bias_res_deep_conv1_expand
        expand_conv_bias_with_param(m.bias_res_deep_conv3_stack_list[i][0], new_m.bias_res_deep_conv3_stack_list[i][0])

        # res_deep_conv3
        expand_conv_weight_with_param(m.res_deep_conv3_stack_list[i][1], new_m.res_deep_conv3_stack_list[i][1])
        # bias_res_deep_conv3
        expand_conv_bias_with_param(m.bias_res_deep_conv3_stack_list[i][1], new_m.bias_res_deep_conv3_stack_list[i][1])

        # res_deep_conv1_squeeze
        expand_conv_weight_with_param(m.res_deep_conv3_stack_list[i][2], new_m.res_deep_conv3_stack_list[i][2])
        # bias_res_deep_conv1_squeeze
        expand_conv_bias_with_param(m.bias_res_deep_conv3_stack_list[i][2], new_m.bias_res_deep_conv3_stack_list[i][2])
    # scale
    old_inplane = getattr(m, 'scale').shape[1]
    new_inplane = getattr(new_m, 'scale').shape[1]
    expand_inplane_ratio = new_inplane // old_inplane
    # init new weight
    new_weight = getattr(m, 'scale').data
    if expand_inplane_ratio > 1:
        new_weight = torch.cat(
            [new_weight + noise_scale * torch.normal(0, 1e-6, size=new_weight.shape) for i in range(expand_inplane_ratio)], 1)
    getattr(new_m, 'scale').data.copy_(new_weight)

# static2dynamic and dynamic2static
def recursive_hasattr(model, attr_name):
    output = False
    if len(attr_name) == 0:
        return True
    module_name = attr_name.split(".")[0]
    if hasattr(model, module_name):
        output = recursive_hasattr(getattr(model, module_name), attr_name[len(module_name) + 1:])
    else:
        output = False
    return output


def recursive_setattr(model, attr_name, value):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if len(attr_name_split) == 1:
        setattr(model, module_name, value)
        return
    recursive_setattr(getattr(model, module_name), attr_name[len(module_name) + 1:], value)


def recursive_getattr(model, attr_name):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if len(attr_name_split) == 1:
        value = None
        source_model = getattr(model, module_name)
        return source_model
    return recursive_getattr(getattr(model, module_name), attr_name[len(module_name) + 1:])


def static2dynamic_single(model, attr_name, whole_module_name, gpd_ratio, build_expand_ratio, is_first_module, is_last_module):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    build_expand_ratio_in = 1 if is_first_module else build_expand_ratio
    build_expand_ratio_out = 1 if is_last_module else build_expand_ratio
    if len(attr_name_split) == 1:
        value = None
        source_model = getattr(model, module_name)
        if isinstance(source_model, nn.Conv2d):
            new_groups = source_model.groups
            if build_expand_ratio > 1:
                if new_groups > 1:
                    new_groups = new_groups * build_expand_ratio
            value = DynamicConv2d(
                source_model.in_channels * build_expand_ratio_in,
                source_model.out_channels * build_expand_ratio_out,
                source_model.kernel_size[0],
                stride=source_model.stride,
                padding=source_model.padding,
                dilation=source_model.dilation,
                groups=new_groups,
                bias=source_model.bias is not None,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, nn.Linear):
            value = DynamicLinear(
                source_model.in_features * build_expand_ratio_in,
                source_model.out_features * build_expand_ratio_out,
                bias=source_model.bias is not None,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, nn.ConvTranspose2d):
            new_groups = source_model.groups
            if build_expand_ratio > 1:
                if new_groups > 1:
                    new_groups = new_groups * build_expand_ratio
            value = DynamicConvTranspose2d(
                source_model.in_channels * build_expand_ratio_in,
                source_model.out_channels * build_expand_ratio_out,
                source_model.kernel_size[0],
                stride=source_model.stride,
                padding=source_model.padding,
                output_padding=source_model.output_padding,
                groups=new_groups,
                bias=source_model.bias is not None,
                dilation=source_model.dilation,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, nn.BatchNorm2d):
            value = DynamicBatchNorm2d(
                source_model.num_features * build_expand_ratio_out,
                # source_model.eps,
                momentum = source_model.momentum,
                affine = source_model.affine,
                track_running_stats = source_model.track_running_stats,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict(), strict=False)
        elif isinstance(source_model, nn.PReLU):
            value = DynamicPReLU(
                source_model.num_parameters * build_expand_ratio_out,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, DotProduct):
            value = DynamicDotProduct(
                source_model.num_features * build_expand_ratio_out,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
        elif isinstance(source_model, scale_module): # only support single dim larger than 1
            new_shape = source_model.weight.shape
            valid_dim = 0
            for i in range(len(new_shape)):
                if new_shape[i] > 1:
                    new_shape[i] = new_shape[i] * build_expand_ratio_out
                    valid_dim = i
                    break
            value = DynamicScale(
                new_shape, valid_dim
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, OREPA):
            new_groups = source_model.groups
            if build_expand_ratio > 1:
                if new_groups > 1:
                    new_groups = new_groups * build_expand_ratio
            value = DynamicOREPA(
                source_model.in_channels * build_expand_ratio_in, source_model.out_channels * build_expand_ratio_out, source_model.kernel_size, source_model.stride, source_model.padding, source_model.dilation, new_groups, source_model.internal_channels_1x1_3x3, source_model.deploy, source_model.nonlinear, source_model.single_init, source_model.weight_only, source_model.single_branch_preserve, source_model.init_hyper_para, source_model.init_hyper_gamma, in_channels_expanded=source_model.in_channels_expanded if hasattr(source_model, 'in_channels_expanded') else False, out_channels_expanded=source_model.out_channels_expanded if hasattr(source_model, 'out_channels_expanded') else False
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, OREPA_SE_4x):
            value = DynamicOREPA_SE_4x(
                source_model.in_channels * build_expand_ratio_in, source_model.out_channels * build_expand_ratio_out, source_model.kernel_size, source_model.stride, source_model.padding, source_model.dilation, source_model.groups, source_model.internal_channels_1x1_3x3, source_model.deploy, source_model.nonlinear, source_model.single_init, source_model.weight_only, source_model.single_branch_preserve, source_model.init_hyper_para, source_model.init_hyper_gamma, in_channels_expanded=source_model.in_channels_expanded if hasattr(source_model, 'in_channels_expanded') else False, out_channels_expanded=source_model.out_channels_expanded if hasattr(source_model, 'out_channels_expanded') else False
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, TokenMixing):
            value = DynamicTokenMixing(
                dim=source_model.dim, num_heads=source_model.num_heads, qkv_bias=source_model.qkv_bias, qk_scale=source_model.qk_scale, attn_drop=source_model.attn_drop_prob, proj_drop=source_model.proj_drop_prob, sr_ratio=source_model.sr_ratio, linear=source_model.linear, share_atten=source_model.share_atten, emlp=source_model.emlp
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, nn.LayerNorm):
            value = DynamicLayerNorm(
                source_model.normalized_shape[0] * build_expand_ratio_out,
                eps=source_model.eps,
                elementwise_affine=source_model.elementwise_affine,
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, nn.GELU):
            try:
                value = DynamicGELU(
                    source_model.approximate
                )
            except:
                value = DynamicGELU()
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
        elif isinstance(source_model, ChannelProcessing):
            value = DynamicChannelProcessing(
                dim=source_model.dim, num_heads=source_model.num_heads, qkv_bias=source_model.qkv_bias, attn_drop=source_model.attn_drop_prob, linear=source_model.linear, drop_path=source_model.drop_path_prob, mlp_hidden_dim=source_model.mlp_hidden_dim, act_layer=source_model.act_layer, drop=source_model.drop, norm_layer=source_model.norm_layer, cha_sr_ratio=source_model.cha_sr_ratio, c_head_num=source_model.c_head_num
            )
            contain_orepa = check_contain_orepa(source_model)
            if contain_orepa:
                value = transfer2orepa(value, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, ConvNeXtBlock):
            value = DynamicConvNeXtBlock(
                dim=source_model.dim, drop_path=source_model.drop_path_prob, ls_init_value=source_model.ls_init_value, conv_mlp=source_model.conv_mlp, mlp_ratio=source_model.mlp_ratio, norm_layer=source_model.norm_layer
            )
            contain_orepa = check_contain_orepa(source_model)
            if contain_orepa:
                value = transfer2orepa(value, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, FANBlock):
            value = DynamicFANBlock(
                dim=source_model.dim, num_heads=source_model.num_heads, mlp_ratio=source_model.mlp_ratio, qkv_bias=source_model.qkv_bias, drop=source_model.drop, attn_drop=source_model.attn_drop, sharpen_attn=source_model.sharpen_attn, drop_path=source_model.drop_path_prob, act_layer=source_model.act_layer, norm_layer=source_model.norm_layer, eta=source_model.eta, sr_ratio=source_model.sr_ratio, downsample=source_model.downsample, c_head_num=source_model.c_head_num
            )
            contain_orepa = check_contain_orepa(source_model)
            if contain_orepa:
                value = transfer2orepa(value, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            value.H = source_model.H
            value.W = source_model.W
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, ClassAttn):
            value = DynamicClassAttn(
                dim=source_model.dim, num_heads=source_model.num_heads, qkv_bias=source_model.qkv_bias, attn_drop=source_model.attn_drop_prob, proj_drop=source_model.proj_drop_prob
            )
            contain_orepa = check_contain_orepa(source_model)
            if contain_orepa:
                value = transfer2orepa(value, train_from_scratch=True, print_info=False, contain_orepa=contain_orepa)
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        elif isinstance(source_model, Attention):
            value = DynamicAttention(
                dim=source_model.dim, num_heads=source_model.num_heads, qkv_bias=source_model.qkv_bias, qk_scale=source_model.qk_scale, attn_drop=source_model.attn_drop_prob, proj_drop=source_model.proj_drop_prob, use_mask=source_model.use_mask
            )
            value.gpd_ratio = gpd_ratio
            value.module_name = whole_module_name
            if build_expand_ratio == 1:
                value.load_state_dict(source_model.state_dict())
        setattr(model, module_name, value)
        return
    static2dynamic_single(getattr(model, module_name), attr_name[len(module_name) + 1:], attr_name, gpd_ratio,
                          build_expand_ratio, is_first_module, is_last_module)


def static2dynamic(tiny_model, model, build_expand_ratio=1, first_module_name_list=None, last_module_name_list=None, exclude_module_name_list=None):
    new_model = copy.deepcopy(model)
    max_gpd_ratio = 1
    if first_module_name_list is not None:
        for i in range(len(first_module_name_list)):
            first_module_name = first_module_name_list[i]
            first_module_name_list[i] = first_module_name[7:] if first_module_name.startswith('module.') else first_module_name
    if last_module_name_list is not None:
        for i in range(len(last_module_name_list)):
            last_module_name = last_module_name_list[i]
            last_module_name_list[i] = last_module_name[7:] if last_module_name.startswith('module.') else last_module_name
    if exclude_module_name_list is not None:
        for i in range(len(exclude_module_name_list)):
            exclude_module_name = exclude_module_name_list[i]
            exclude_module_name_list[i] = exclude_module_name[7:] if exclude_module_name.startswith('module.') else exclude_module_name

    # tranfer tokenmixing
    for name_module0, name_module1 in zip(tiny_model.named_modules(), new_model.named_modules()):
        name0, m0 = name_module0
        name1, m1 = name_module1
        if isinstance(m1, (TokenMixing, ChannelProcessing, ClassAttn)):
            attr_name = get_outchannels(m1.q)
            gpd_ratio = get_attr_value(getattr(m1.q, attr_name)) // get_attr_value(getattr(m0.q, attr_name))
            static2dynamic_single(new_model, name1, name1, gpd_ratio, build_expand_ratio, False, False)
        if isinstance(m1, (ConvNeXtBlock)):
            attr_name = get_outchannels(m1.conv_dw)
            gpd_ratio = get_attr_value(getattr(m1.conv_dw, attr_name)) // get_attr_value(getattr(m0.conv_dw, attr_name))
            static2dynamic_single(new_model, name1, name1, gpd_ratio, build_expand_ratio, False, False)
        if isinstance(m1, (FANBlock)):
            attr_name = get_outchannels(m1.attn.q)
            gpd_ratio = get_attr_value(getattr(m1.attn.q, attr_name)) // get_attr_value(getattr(m0.attn.q, attr_name))
            static2dynamic_single(new_model, name1, name1, gpd_ratio, build_expand_ratio, False, False)
        if isinstance(m1, (Attention)):
            attr_name = get_outchannels(m1.qkv)
            gpd_ratio = get_attr_value(getattr(m1.qkv, attr_name)) // get_attr_value(getattr(m0.qkv, attr_name))
            static2dynamic_single(new_model, name1, name1, gpd_ratio, build_expand_ratio, False, False)

    # transfer other types
    for name_module0, name_module1 in zip(tiny_model.named_modules(), new_model.named_modules()):
        name0, m0 = name_module0
        name1, m1 = name_module1

        if exclude_module_name_list is not None:
            is_excluded = False
            for exclude_module_name in exclude_module_name_list:
                if name0.startswith(exclude_module_name):
                    is_excluded = True
                    break
            if is_excluded:
                continue

        is_first_module = False
        if build_expand_ratio > 1:
            if first_module_name_list is not None:
                for first_module_name in first_module_name_list:
                    if name1 == first_module_name:
                        is_first_module = True
                        break

        is_last_module = False
        if build_expand_ratio > 1:
            if last_module_name_list is not None:
                for last_module_name in last_module_name_list:
                    if name1 == last_module_name:
                        is_last_module = True
                        break

        if isinstance(m1, static_type) and not isinstance(m1, TokenMixing) and not isinstance(m1, nn.GELU) and not isinstance(m1, ChannelProcessing) and not isinstance(m1, ConvNeXtBlock) and not isinstance(m1, FANBlock) and not isinstance(m1, ClassAttn) and not isinstance(m1, Attention):
            attr_name = get_outchannels(m1)
            gpd_ratio = get_attr_value(getattr(m1, attr_name)) // get_attr_value(getattr(m0, attr_name))
            max_gpd_ratio = max(max_gpd_ratio, gpd_ratio)
            static2dynamic_single(new_model, name1, name1, gpd_ratio, build_expand_ratio, is_first_module,
                                  is_last_module)
        if isinstance(m1, nn.GELU):
            static2dynamic_single(new_model, name1, name1, max_gpd_ratio, build_expand_ratio, is_first_module,
                                  is_last_module)

    # set gpd_ratio for dynamic tokenmixing
    for name_module0, name_module1 in zip(tiny_model.named_modules(), new_model.named_modules()):
        name0, m0 = name_module0
        name1, m1 = name_module1
        if isinstance(m1, (DynamicTokenMixing, DynamicClassAttn)) and m1.gpd_ratio==1:
            attr_name = get_outchannels(m1.q)
            gpd_ratio = get_attr_value(getattr(m1.q, attr_name)) // get_attr_value(getattr(m0.q, attr_name))
            m1.gpd_ratio = gpd_ratio
            m1.dim = gpd_ratio * m1.dim
            m1.num_heads = gpd_ratio * m1.num_heads
        if isinstance(m1, DynamicChannelProcessing) and m1.gpd_ratio==1:
            attr_name = get_outchannels(m1.q)
            gpd_ratio = get_attr_value(getattr(m1.q, attr_name)) // get_attr_value(getattr(m0.q, attr_name))
            m1.gpd_ratio = gpd_ratio
            m1.dim = gpd_ratio * m1.dim
            m1.num_heads = gpd_ratio * m1.num_heads
            m1.temperature = nn.Parameter(torch.ones(m1.temperature.shape[0] * gpd_ratio, 1, 1))
            m1.mlp_v.gamma = nn.Parameter(torch.ones(m1.mlp_v.gamma.shape[0] * gpd_ratio), requires_grad=True)
        if isinstance(m1, DynamicConvNeXtBlock) and m1.gpd_ratio==1:
            attr_name = get_outchannels(m1.conv_dw)
            gpd_ratio = get_attr_value(getattr(m1.conv_dw, attr_name)) // get_attr_value(getattr(m0.conv_dw, attr_name))
            m1.gpd_ratio = gpd_ratio
            m1.dim = gpd_ratio * m1.dim
            m1.gamma = nn.Parameter(m1.ls_init_value * torch.ones(m1.gamma.shape[0] * gpd_ratio))
        if isinstance(m1, DynamicFANBlock) and m1.gpd_ratio==1:
            attr_name = get_outchannels(m1.attn.q)
            gpd_ratio = get_attr_value(getattr(m1.attn.q, attr_name)) // get_attr_value(getattr(m0.attn.q, attr_name))
            m1.gpd_ratio = gpd_ratio
            m1.dim = gpd_ratio * m1.dim
            m1.num_heads = gpd_ratio * m1.num_heads
            m1.gamma1 = nn.Parameter(m1.eta * torch.ones(m1.gamma1.shape[0] * gpd_ratio), requires_grad=True)
            m1.gamma2 = nn.Parameter(m1.eta * torch.ones(m1.gamma2.shape[0] * gpd_ratio), requires_grad=True)
        if isinstance(m1, ClassAttentionBlock) and build_expand_ratio > 1:
            if not hasattr(m1, 'expand_gamma1') or not m1.expand_gamma1:
                m1.expand_gamma1 = True
                m1.gamma1 = nn.Parameter(m1.eta * torch.ones(m1.dim * build_expand_ratio), requires_grad=True)
            if not hasattr(m1, 'expand_gamma2') or not m1.expand_gamma2:
                m1.expand_gamma2 = True
                m1.gamma2 = nn.Parameter(m1.eta * torch.ones(m1.dim * build_expand_ratio), requires_grad=True)
            m1.dim = m1.gamma1.shape[0]
        if isinstance(m1, (DynamicAttention)) and m1.gpd_ratio==1:
            attr_name = get_outchannels(m1.qkv)
            gpd_ratio = get_attr_value(getattr(m1.qkv, attr_name)) // get_attr_value(getattr(m0.qkv, attr_name))
            m1.gpd_ratio = gpd_ratio
            m1.dim = gpd_ratio * m1.dim
            m1.num_heads = gpd_ratio * m1.num_heads

    if isinstance(new_model, FAN) and build_expand_ratio > 1:
        if not hasattr(new_model, 'expand_cls_token') or not new_model.expand_cls_token:
            new_model.expand_cls_token = True
            new_model.cls_token = nn.Parameter(torch.zeros(1, 1, new_model.cls_token.shape[2] * build_expand_ratio))

    return new_model


def dynamic2static_single(model, attr_name, all_types):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if len(attr_name_split) == 1:
        source_model = getattr(model, module_name)
        if isinstance(source_model, all_types):
            value = source_model.export()
            setattr(model, module_name, value)
        return
    dynamic2static_single(getattr(model, module_name), attr_name[len(module_name)+1:], all_types)


def dynamic2static(model, my_dynamic_type=None, my_fake_quantizer=None):
    new_model = copy.deepcopy(model)
    active_ratio = 1
    for name, m in new_model.named_modules():
        # if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d, nn.BatchNorm2d)):
        all_types = dynamic_type
        # DynamicLearnableFakeQuantize
        all_types = all_types + my_dynamic_type if my_dynamic_type is not None else all_types # incorporate dopt conv layers
        all_types = all_types + my_fake_quantizer if my_fake_quantizer is not None else all_types # incorporate dopt fake quantizers
        # TODO
        # if isinstance(m, all_types) and not isinstance(m, (ChannelProcessing, ConvNeXtBlock, FANBlock)):
        if isinstance(m, all_types):
            dynamic2static_single(new_model, name, all_types)

        if isinstance(m, ClassAttentionBlock):
            if hasattr(m, 'expand_gamma1') and m.expand_gamma1:
                active_ratio = m.attn.active_ratio
                if active_ratio > 1:
                    m.expand_gamma1 = False
                new_gamma1 = m.gamma1[:m.gamma1.shape[0] // active_ratio]
                new_gamma1 = avg_param_for_tiny_model1d(new_gamma1, m.gamma1, m.gamma1.shape[0] // active_ratio)
                new_gamma1 = nn.Parameter(new_gamma1)
                m.gamma1 = new_gamma1
            if hasattr(m, 'expand_gamma2') and m.expand_gamma2:
                active_ratio = m.attn.active_ratio
                if active_ratio > 1:
                    m.expand_gamma2 = False
                new_gamma2 = m.gamma2[:m.gamma2.shape[0] // active_ratio]
                new_gamma2 = avg_param_for_tiny_model1d(new_gamma2, m.gamma2, m.gamma2.shape[0] // active_ratio)
                new_gamma2 = nn.Parameter(new_gamma2)
                m.gamma2 = new_gamma2

    if isinstance(new_model, FAN):
        if hasattr(new_model, 'expand_cls_token') and new_model.expand_cls_token:
            if active_ratio > 1:
                new_model.expand_cls_token = False
            # gpd_ratio = get_gpd_ratio(new_model)
            new_cls_token = new_model.cls_token[:, :, : new_model.cls_token.shape[2] // active_ratio]
            new_cls_token = avg_param_for_tiny_model1d(new_cls_token, new_model.cls_token, new_model.cls_token.shape[2] // active_ratio)
            new_cls_token = nn.Parameter(new_cls_token)
            new_model.cls_token = new_cls_token

    return new_model


def build_expanded_net(tiny_model, first_module_name_list=[], last_module_name_list=[], gpd_ratio=2, exclude_module_name_list=[], test_input=None):
    # tiny_model = replace_scale(tiny_model)
    expanded_model = static2dynamic(tiny_model, tiny_model, build_expand_ratio=gpd_ratio, first_module_name_list=first_module_name_list, last_module_name_list=last_module_name_list, exclude_module_name_list=exclude_module_name_list)
    expanded_model.eval()
    if test_input is not None:
        # _ = forward_tiny(expanded_model, test_input)
        _ = forward_expanded(expanded_model, test_input)
    expanded_model = dynamic2static(expanded_model)
    return expanded_model


# def quant_dynamic2static(model):
#     model = dynamic2static(model)
#     # activation quant mapping
#     for name, m in model.named_modules():
#         if isinstance(m, DynamicLearnableFakeQuantize):
#             if m.ch_axis < 0:
#                 scale_shape = m.scale.data.shape
#                 if len(scale_shape) == 1:
#                     pass
#                 else:
#                     m.scale.data = torch.chunk(m.scale.data, m.gpd_ratio)[0] * m.gpd_ratio
#                     m.zero_point.data = torch.chunk(m.zero_point.data, m.gpd_ratio)[0] * m.gpd_ratio
#
#     return model


def get_modules_recursively(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_modules_recursively(child))
            except TypeError:
                flatt_children.append(get_modules_recursively(child))
    return flatt_children


def replace_scale(model):
    contain_OREPA_SE_4x = False
    for name_module, name_module in model.named_modules():
        if isinstance(name_module, OREPA_SE_4x):
            contain_OREPA_SE_4x = True

    contain_ChannelProcessing = False
    for name_module, name_module in model.named_modules():
        if isinstance(name_module, ChannelProcessing):
            contain_ChannelProcessing = True

    if not contain_OREPA_SE_4x and not contain_ChannelProcessing:
        return model

    for module in model.modules():
        if module._get_name() == 'rep_vgg_block_v1_rs':
            if hasattr(module, 'scale') and isinstance(module.scale, (Parameter)):
                source_model = module.scale
                value = scale_module(
                    source_model.shape[1],
                )
                with torch.no_grad():
                    value.weight.copy_(source_model)
                del module.scale
                setattr(module, 'scale', value)
        if isinstance(module, ChannelProcessing):
            if hasattr(module, 'temperature') and isinstance(module.temperature, (Parameter)):
                new_shape = module.temperature.shape
                valid_dim = 0
                for i in range(len(new_shape)):
                    if new_shape[i] > 1:
                        valid_dim = i
                        break
                source_model = module.temperature
                value = scale_module(
                    new_shape, valid_dim
                )
                with torch.no_grad():
                    value.weight.copy_(source_model)
                del module.temperature
                setattr(module, 'temperature', value)
            if hasattr(module.mlp_v, 'gamma') and isinstance(module.mlp_v.gamma, (Parameter)):
                new_shape = module.mlp_v.gamma.shape
                valid_dim = 0
                for i in range(len(new_shape)):
                    if new_shape[i] > 1:
                        valid_dim = i
                        break
                source_model = module.mlp_v.gamma
                value = scale_module(
                    new_shape, valid_dim
                )
                with torch.no_grad():
                    value.weight.copy_(source_model)
                del module.mlp_v.gamma
                setattr(module.mlp_v, 'gamma', value)
    return model


def recover_parameter(model):
    for module in model.modules():
        if module._get_name() == 'rep_vgg_block_v1_rs':
            if isinstance(module.scale, (scale_module)):
                source_model = module.scale.weight
                value = nn.Parameter(
                    torch.Tensor(source_model.shape)
                )
                with torch.no_grad():
                    value.copy_(source_model)
                del module.scale
                setattr(module, 'scale', value)
    return model


def init_expand_param_with_name(tiny_object_model, object_model, attr_name, use_scale=False):
    new_shape = getattr(tiny_object_model, attr_name).shape
    valid_dim = 0
    for i in range(len(new_shape)):
        if new_shape[i] > 1:
            valid_dim = i
            break
    # old config
    old_outplane = getattr(tiny_object_model, attr_name).shape[valid_dim]
    # new config
    new_outplane = getattr(object_model, attr_name).shape[valid_dim]
    expand_outplane_ratio = new_outplane // old_outplane
    # init new weight
    new_weight = getattr(tiny_object_model, attr_name).data
    if expand_outplane_ratio > 1:
        if use_scale:
            new_weight = torch.cat([new_weight / expand_outplane_ratio for i in range(expand_outplane_ratio)], valid_dim)
        else:
            new_weight = torch.cat([new_weight for i in range(expand_outplane_ratio)], valid_dim)
    getattr(object_model, attr_name).data.copy_(new_weight)


def expand_init_wide_model(tiny_object_model, object_model, load_pretrained=True):
    # replace nn.Parameter with scale_module
    if load_pretrained:
        if isinstance(object_model, FAN):
            if hasattr(object_model, 'expand_cls_token') and object_model.expand_cls_token:
                init_expand_param_with_name(tiny_object_model, object_model, 'cls_token')

        for name_module0, name_module1 in zip(tiny_object_model.named_modules(), object_model.named_modules()):
            name0, module0 = name_module0
            name1, module1 = name_module1
        # for module0, module1 in zip(tiny_object_model.named_modules, object_model.named_modules()):
            if isinstance(module0, (nn.Conv2d, nn.Linear)):
                init_expand_module(module0, module1)
            if isinstance(module0, nn.ConvTranspose2d):
                init_expand_deconv_module(module0, module1)
            if isinstance(module0, nn.BatchNorm2d):
                init_expand_bn(module0, module1)
            if isinstance(module0, nn.PReLU):
                init_expand_prelu(module0, module1)
            if isinstance(module0, scale_module):
                init_expand_scale(module0, module1)
            if isinstance(module0, OREPA):
                init_expand_orepa(module0, module1)
            if isinstance(module0, OREPA_SE_4x):
                init_expand_orepa_se_4x(module0, module1)
            if isinstance(module0, nn.LayerNorm):
                init_expand_ln(module0, module1)
            if isinstance(module0, ChannelProcessing):
                # init_expand_channelprocessing(module0, module1)
                init_expand_param_with_name(module0, module1, 'temperature')
                init_expand_param_with_name(module0.mlp_v, module1.mlp_v, 'gamma')
            if isinstance(module0, ConvNeXtBlock):
                # init_expand_convnextblock(module0, module1)
                init_expand_param_with_name(module0, module1, 'gamma')
            if isinstance(module0, FANBlock):
                # init_expand_fanblock(module0, module1)
                init_expand_param_with_name(module0, module1, 'gamma1')
                init_expand_param_with_name(module0, module1, 'gamma2')
            if isinstance(module0, ClassAttentionBlock):
                if hasattr(module1, 'expand_gamma1') and module1.expand_gamma1:
                    init_expand_param_with_name(module0, module1, 'gamma1')
                if hasattr(module1, 'expand_gamma2') and module1.expand_gamma2:
                    init_expand_param_with_name(module0, module1, 'gamma2')
    return object_model


def set_state4orepa(tiny_object_model, object_model):
    for name_module0, name_module1 in zip(tiny_object_model.named_modules(), object_model.named_modules()):
        name0, module0 = name_module0
        name1, module1 = name_module1
        if isinstance(module0, (OREPA, OREPA_SE_4x)):
            # set in_out_simult_expanded
            m_in_channels, m_out_channels = module0.in_channels, module0.out_channels
            new_m_in_channels, new_m_out_channels = module1.in_channels, module1.out_channels
            if new_m_in_channels > m_in_channels:
                setattr(module1, 'in_channels_expanded', True)
            if new_m_out_channels > m_out_channels:
                setattr(module1, 'out_channels_expanded', True)
    return object_model



def expand_init_wide_model_anysize(tiny_object_model, object_model):
    # replace nn.Parameter with scale_module
    # tiny_object_model = replace_scale(tiny_object_model)
    # object_model = replace_scale(object_model)
    for module0, module1 in zip(tiny_object_model.modules(), object_model.modules()):
        if isinstance(module0, (nn.Conv2d, nn.Linear)):
            init_expand_module_anysize(module0, module1)
        if isinstance(module0, (nn.ConvTranspose2d)):
            init_expand_deconv_module_anysize(module0, module1)
        if isinstance(module0, (nn.BatchNorm2d)):
            init_expand_bn_anysize(module0, module1)
        if isinstance(module0, (nn.PReLU)):
            init_expand_prelu_anysize(module0, module1)
        if isinstance(module0, (scale_module)):
            init_expand_paramater_anysize(module0, module1)
    return object_model

def init_tiny_module(m, new_m):
    # old config
    old_outplane, old_inplane = m.weight.shape[0], m.weight.shape[1]
    old_bias = m.bias
    # new config
    new_outplane, new_inplane = new_m.weight.shape[0], new_m.weight.shape[1]
    new_bias = new_m.bias
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = new_m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.split(new_weight * expand_outplane_ratio, [old_outplane] * expand_outplane_ratio, dim=0)[0]  # TODO: check
    if expand_inplane_ratio > 1:
        new_weight = torch.split(new_weight, [old_inplane] * expand_inplane_ratio, dim=1)[0]
    m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        assert expand_outplane_ratio * old_bias.shape[0] == new_bias.shape[
            0], f'old_bias {expand_outplane_ratio * old_bias.shape[0]} has different shape with new_bias {new_bias.shape[0]}'
        new_bias = new_m.bias.data
        if expand_outplane_ratio > 1:
            new_bias = torch.split(new_bias * expand_outplane_ratio, [old_outplane] * expand_outplane_ratio, dim=0)[0]
        m.bias.data.copy_(new_bias)


def init_tiny_deconv_module(m, new_m):
    # old config
    old_inplane, old_outplane = m.weight.shape[0], m.weight.shape[1]
    old_bias = m.bias
    # new config
    new_inplane, new_outplane = new_m.weight.shape[0], new_m.weight.shape[1]
    new_bias = new_m.bias
    expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    # init new weight
    new_weight = new_m.weight.data
    if expand_outplane_ratio > 1:
        new_weight = torch.split(new_weight * expand_outplane_ratio, [old_outplane] * expand_outplane_ratio, dim=1)[0]
    if expand_inplane_ratio > 1:
        new_weight = torch.split(new_weight, [old_inplane] * expand_inplane_ratio, dim=0)[0]
    m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        assert expand_outplane_ratio * old_bias.shape[0] == new_bias.shape[
            0], f'old_bias {expand_outplane_ratio * old_bias.shape[0]} has different shape with new_bias {new_bias.shape[0]}'
        new_bias = new_m.bias.data
        if expand_outplane_ratio > 1:
            new_bias = torch.split(new_bias * expand_outplane_ratio, [old_outplane] * expand_outplane_ratio, dim=0)[0]
        m.bias.data.copy_(new_bias)

def squeeze_2_tiny_model(tiny_object_model, object_model):
    for module0, module1 in zip(tiny_object_model.modules(), object_model.modules()):
        if isinstance(module0, (nn.Conv2d, nn.Linear)):
            print("squeezing {}".format(module0))
            init_tiny_module(module0, module1)
        if isinstance(module0, (nn.ConvTranspose2d)):
            print("squeezing {}".format(module0))
            init_tiny_deconv_module(module0, module1)
        # if isinstance(module0, (nn.BatchNorm2d)):
        #     print("squeezing {}".format(module0))
        #     init_tiny_bn(module0, module1)
    return tiny_object_model


def set_active_with_ratio_rescale(model, ratio=1):
    target_model = model.module if hasattr(model, 'module') else model
    for m in target_model.modules():
        if isinstance(m, dynamic_type):
            if m.gpd_ratio > 1:
                if not isinstance(m, (DynamicTokenMixing, DynamicLayerNorm, DynamicGELU, DynamicChannelProcessing, DynamicConvNeXtBlock, DynamicFANBlock, DynamicClassAttn, DynamicAttention)):
                    attr_name = get_outchannels(m)
                    full_out_channels = get_attr_value(getattr(m, attr_name))
                    setattr(m, 'active_' + attr_name, full_out_channels // ratio)
                m.active_ratio = ratio


def quant_static2dynamic(tiny_model, model, my_dynamic_type=None, my_fake_quantizer=None):
    for name_module0, name_module1 in zip(tiny_model.named_modules(), model.named_modules()):
        name0, m0 = name_module0
        name1, m1 = name_module1

        if my_dynamic_type is not None and isinstance(m1, static_type):
            attr_name = get_outchannels(m1)
            gpd_ratio = get_attr_value(getattr(m1, attr_name)) // get_attr_value(getattr(m0, attr_name))
            static2dynamic_single(model, name1, gpd_ratio)

    return model


def insert_dynamic_batchnorm(tiny_model, model):
    for name_module0, name_module1 in zip(tiny_model.named_modules(), model.named_modules()):
        name0, m0 = name_module0
        name1, m1 = name_module1
        if isinstance(m1, nn.BatchNorm2d):
            gpd_ratio = m1.num_features // m0.num_features
            static2dynamic_single(model, name1, gpd_ratio)
    return model


def insert_dynamic_dotproduct(tiny_model, model):
    for name_module0, name_module1 in zip(tiny_model.named_modules(), model.named_modules()):
        name0, m0 = name_module0
        name1, m1 = name_module1
        if isinstance(m1, DotProduct):
            gpd_ratio = m1.num_features // m0.num_features
            static2dynamic_single(model, name1, gpd_ratio)
    return model


def reinit_fake_quantize(model):
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            submodule.scale.data = torch.ones_like(submodule.scale)
            submodule.zero_point.data = torch.zeros_like(submodule.zero_point.float())


# def copy_scale_zeropoint_2expanded(tiny_model, model):
#     for name_module0, name_module1 in zip(tiny_model.named_modules(), model.named_modules()):
#         name0, m0 = name_module0
#         name1, m1 = name_module1
#
#         if isinstance(m1, DynamicLearnableFakeQuantize):
#             expand_outplane_ratio = m1.x_channels // m0.x_channels
#             # copy scale from tiny to large
#             new_scale = m0.scale.data
#             if expand_outplane_ratio > 1:
#                 if m1.scale.shape[0] > 1:
#                     new_scale = torch.cat([new_scale / expand_outplane_ratio for i in range(expand_outplane_ratio)], 0)
#                 else:
#                     new_scale = new_scale / expand_outplane_ratio
#             m1.scale.data.copy_(new_scale)
#             # copy zero_point from tiny to large
#             new_zero_point = m0.zero_point.data
#             if expand_outplane_ratio > 1 and m1.scale.shape[0] > 1:
#                 new_zero_point = torch.cat([new_zero_point for i in range(expand_outplane_ratio)], 0)
#             m1.zero_point.data.copy_(new_zero_point)


# def copy_scale_zeropoint_2expanded_partial(tiny_model, model):
#     for name_module0, name_module1 in zip(tiny_model.named_modules(), model.named_modules()):
#         name0, m0 = name_module0
#         name1, m1 = name_module1
#
#         if isinstance(m1, DynamicLearnableFakeQuantize):
#             expand_outplane_ratio = m1.x_channels // m0.x_channels
#             # copy scale from tiny to large
#             new_scale = m0.scale.data
#             large_scale = m1.scale.data
#             if expand_outplane_ratio > 1:
#                 if m1.scale.shape[0] > 1:
#                     new_scale = torch.cat([new_scale / expand_outplane_ratio, large_scale[m0.x_channels:]], 0)
#                 else:
#                     new_scale = new_scale / expand_outplane_ratio
#             m1.scale.data.copy_(new_scale)
#             # copy zero_point from tiny to large
#             new_zero_point = m0.zero_point.data
#             if expand_outplane_ratio > 1 and m1.scale.shape[0] > 1:
#                 new_zero_point = torch.cat([new_zero_point for i in range(expand_outplane_ratio)], 0)
#             m1.zero_point.data.copy_(new_zero_point)


# def copy_scale_zeropoint_2expanded_self(tiny_model, model):
#     for name_module0, name_module1 in zip(tiny_model.named_modules(), model.named_modules()):
#         name0, m0 = name_module0
#         name1, m1 = name_module1
#
#         if isinstance(m1, DynamicLearnableFakeQuantize):
#             expand_outplane_ratio = m1.x_channels // m0.x_channels
#             # copy scale from tiny to large
#             new_scale = m1.scale.data
#             if expand_outplane_ratio > 1:
#                 if m1.scale.shape[0] > 1:
#                     new_scale = torch.cat([new_scale[: m1.scale.shape[0] // expand_outplane_ratio] / expand_outplane_ratio for i in range(expand_outplane_ratio)], 0)
#                 else:
#                     new_scale = new_scale / expand_outplane_ratio
#             m1.scale.data.copy_(new_scale)
#             # copy zero_point from tiny to large
#             new_zero_point = m1.zero_point.data
#             if expand_outplane_ratio > 1 and m1.scale.shape[0] > 1:
#                 new_zero_point = torch.cat([new_zero_point[: m1.scale.shape[0] // expand_outplane_ratio] for i in range(expand_outplane_ratio)], 0)
#             m1.zero_point.data.copy_(new_zero_point)


def quant_set_active_with_ratio_rescale(model, ratio=1, skip_prob=0., skip_prob_tiny=0., my_dynamic_type=None, my_fake_quantizer=None):
    target_model = model.module if hasattr(model, 'module') else model
    for name_module, m in target_model.named_modules():
    # for m in target_model.modules():
        if isinstance(m, dynamic_type if my_dynamic_type is None else my_dynamic_type + dynamic_type):
            attr_name = get_outchannels(m)
            full_out_channels = getattr(m, attr_name)
            if m.gpd_ratio > 1:
                setattr(m, 'active_' + attr_name, full_out_channels // ratio)
                m.active_ratio = ratio
        # if isinstance(m, DynamicLearnableFakeQuantize if my_fake_quantizer is None else my_fake_quantizer):
        #     full_out_channels = m.scale.shape[0]
        #     if m.gpd_ratio > 1:
        #         if full_out_channels > 1:
        #             m.active_out_channels = full_out_channels // ratio
        #         m.active_ratio = ratio



def copy_params_to_tiny(tiny_model, large_model):
    for tiny_p, large_p in zip(tiny_model.named_parameters(), large_model.named_parameters()):
        tiny_name, tiny_param = tiny_p[0], tiny_p[1]
        large_name, large_param = large_p[0], large_p[1]
        assert tiny_name == '.'.join(large_name.split(".")[1:]), "models are not identical"

        if tiny_param.shape == large_param.shape:
            # print("copying {} to {}".format(large_name, tiny_name))
            tiny_p[1].data = large_p[1].data
            continue
        num_dim = len(tiny_param.shape)
        tiny_shape, large_shape = list(tiny_param.size()), list(large_param.size())
        # print(num_dim, tiny_shape, large_shape)
        if num_dim == 4 and (tiny_shape[0] == large_shape[0] // 2 and tiny_shape[1] == large_shape[1] // 2):
            # print("copying {} to {}".format(large_name, tiny_name))
            tiny_p[1].data = large_p[1].data[:large_shape[0]//2, :large_shape[1]//2, ...]
        elif num_dim == 4 and (tiny_shape[0] == large_shape[0] // 2):
            # print("copying {} to {}".format(large_name, tiny_name))
            tiny_p[1].data = large_p[1].data[:large_shape[0]//2, ...]
        elif num_dim == 2 and (tiny_shape[0] == large_shape[0] // 2 and tiny_shape[1] == large_shape[1] // 2):
            # print("copying {} to {}".format(large_name, tiny_name))
            tiny_p[1].data = large_p[1].data[:large_shape[0]//2, :large_shape[1]//2]
        elif num_dim == 2 and (tiny_shape[0] == large_shape[0] // 2):
            # print("copying {} to {}".format(large_name, tiny_name))
            tiny_p[1].data = large_p[1].data[:large_shape[0]//2, ...]
        elif num_dim == 1 and (tiny_shape[0] == large_shape[0] // 2):
            # print("copying {} to {}".format(large_name, tiny_name))
            tiny_p[1].data = large_p[1].data[:large_shape[0] // 2, ...]
        else:
            print("When handling the {} and {} modules, errors".format(tiny_name, large_name))
            raise NotImplementedError


def init_expand_module_anysize(m, new_m):
    # old config
    old_outplane, old_inplane = m.weight.shape[0], m.weight.shape[1]
    old_bias = m.bias
    # new config
    new_outplane, new_inplane = new_m.weight.shape[0], new_m.weight.shape[1]
    new_bias = new_m.bias
    # expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    diff_outplane = new_outplane - old_outplane
    diff_inplane = new_inplane - old_inplane
    # init new weight
    new_weight = m.weight.data
    if diff_outplane > 0:
        new_weight1 = new_weight[:diff_outplane, :, :, :] / 2
        new_weight1_1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight2 = new_weight[diff_outplane:, :, :, :]
        new_weight1_2 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight1_1, new_weight2, new_weight1_2], 0)
    if diff_inplane > 0:
        new_weight1 = new_weight[:, :diff_inplane, :, :]
        new_weight1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight, new_weight1], 1)
    # print(new_m.weight.shape, new_weight.shape)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        new_bias = m.bias.data
        if diff_outplane > 0:
            new_bias1 = new_bias[:diff_outplane] / 2
            new_bias1_1 = new_bias1 + noise_scale * torch.normal(0, 1e-6, size=new_bias1.shape)
            new_bias2 = new_bias[diff_outplane:]
            new_bias1_2 = new_bias1 + noise_scale * torch.normal(0, 1e-6, size=new_bias1.shape)
            new_bias = torch.cat([new_bias1_1, new_bias2, new_bias1_2], 0)
        new_m.bias.data.copy_(new_bias)


def init_expand_deconv_module_anysize(m, new_m):
    # old config
    old_inplane, old_outplane = m.weight.shape[0], m.weight.shape[1]
    old_bias = m.bias
    # new config
    new_inplane, new_outplane = new_m.weight.shape[0], new_m.weight.shape[1]
    new_bias = new_m.bias
    # expand_outplane_ratio, expand_inplane_ratio = new_outplane // old_outplane, new_inplane // old_inplane
    diff_outplane = new_outplane - old_outplane
    diff_inplane = new_inplane - old_inplane

    # init new weight
    new_weight = m.weight.data
    if diff_outplane > 0:
        new_weight1 = new_weight[:, :diff_outplane, :, :] / 2
        new_weight1_1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight2 = new_weight[:, diff_outplane:, :, :]
        new_weight1_2 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight1_1, new_weight2, new_weight1_2], 1)
    if diff_inplane > 0:
        new_weight1 = new_weight[:diff_inplane, :, :, :]
        new_weight1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight, new_weight1], 0)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    if old_bias is not None:
        new_bias = m.bias.data
        if diff_outplane > 0:
            new_bias1 = new_bias[:diff_outplane] / 2
            new_bias1_1 = new_bias1 + noise_scale * torch.normal(0, 1e-6, size=new_bias1.shape)
            new_bias2 = new_bias[diff_outplane:]
            new_bias1_2 = new_bias1 + noise_scale * torch.normal(0, 1e-6, size=new_bias1.shape)
            new_bias = torch.cat([new_bias1_1, new_bias2, new_bias1_2], 0)
        new_m.bias.data.copy_(new_bias)


def init_expand_bn_anysize(m, new_m):
    # old config
    old_outplane = m.weight.shape[0]
    # new config
    new_outplane = new_m.weight.shape[0]
    diff_outplane = new_outplane - old_outplane
    # init new weight
    new_weight = m.weight.data
    if diff_outplane > 0:
        new_weight1 = new_weight[:diff_outplane] / 2
        new_weight1_1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight2 = new_weight[diff_outplane:]
        new_weight1_2 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight1_1, new_weight2, new_weight1_2], 0)
    new_m.weight.data.copy_(new_weight)
    # init new bias
    new_bias = m.bias.data
    if diff_outplane > 0:
        new_bias1 = new_bias[:diff_outplane] / 2
        new_bias1_1 = new_bias1 + noise_scale * torch.normal(0, 1e-6, size=new_bias1.shape)
        new_bias2 = new_bias[diff_outplane:]
        new_bias1_2 = new_bias1 + noise_scale * torch.normal(0, 1e-6, size=new_bias1.shape)
        new_bias = torch.cat([new_bias1_1, new_bias2, new_bias1_2], 0)
    new_m.bias.data.copy_(new_bias)
    # init new running_mean
    new_running_mean = m.running_mean.data
    if diff_outplane > 0:
        new_running_mean1 = new_running_mean[:diff_outplane] / 2
        new_running_mean1_1 = new_running_mean1 + noise_scale * torch.normal(0, 1e-6, size=new_running_mean1.shape)
        new_running_mean2 = new_running_mean[diff_outplane:]
        new_running_mean1_2 = new_running_mean1 + noise_scale * torch.normal(0, 1e-6, size=new_running_mean1.shape)
        new_running_mean = torch.cat([new_running_mean1_1, new_running_mean2, new_running_mean1_2], 0)
    new_m.running_mean.data.copy_(new_running_mean)
    # init new running_mean
    new_running_var = m.running_var.data
    if diff_outplane > 0:
        new_running_var1 = new_running_var[:diff_outplane] / 4
        new_running_var1_1 = new_running_var1 + noise_scale * torch.normal(0, 1e-6, size=new_running_var1.shape)
        new_running_var2 = new_running_var[diff_outplane:]
        new_running_var1_2 = new_running_var1 + noise_scale * torch.normal(0, 1e-6, size=new_running_var1.shape)
        new_running_var = torch.cat([new_running_var1_1, new_running_var2, new_running_var1_2], 0)
    new_m.running_var.data.copy_(new_running_var)
    new_m.eps = m.eps / 4


def init_expand_prelu_anysize(m, new_m):
    # old config
    old_outplane = m.weight.shape[0]
    # new config
    new_outplane = new_m.weight.shape[0]
    diff_outplane = new_outplane - old_outplane
    # init new weight
    new_weight = m.weight.data
    if diff_outplane > 0:
        new_weight1 = new_weight[:diff_outplane]
        new_weight1_1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight2 = new_weight[diff_outplane:]
        new_weight1_2 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight1_1, new_weight2, new_weight1_2], 0)
    new_m.weight.data.copy_(new_weight)


def init_expand_paramater_anysize(m, new_m):
    # old config
    old_outplane = m.weight.shape[1]
    # new config
    new_outplane = new_m.weight.shape[1]
    diff_outplane = new_outplane - old_outplane
    # init new weight
    new_weight = m.weight.data
    if diff_outplane > 0:
        new_weight1 = new_weight[:, :diff_outplane]
        new_weight1_1 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight2 = new_weight[:, diff_outplane:]
        new_weight1_2 = new_weight1 + noise_scale * torch.normal(0, 1e-6, size=new_weight1.shape)
        new_weight = torch.cat([new_weight1_1, new_weight2, new_weight1_2], 1)
    new_m.weight.data.copy_(new_weight)


def get_gpd_ratio(model):
    max_gpd_ratio = 1
    target_model = model.module if hasattr(model, 'module') else model
    for m in target_model.modules():
        if isinstance(m, dynamic_type):
            if m.gpd_ratio > 1:
                gpd_ratio = m.gpd_ratio
                max_gpd_ratio = max(max_gpd_ratio, gpd_ratio)
    return max_gpd_ratio


def gpd_init(tiny_model, first_module_name_list, last_module_name_list, test_input=None, gpd_ratio=2, seed=0, rank=0, exclude_module_name_list=None):
    tiny_model.eval()
    # np.random.seed(seed + rank)
    # torch.manual_seed(seed + rank)
    # torch.cuda.manual_seed(seed + rank)

    # # get full attribute name
    # first_module_name_list = get_full_attr_name(tiny_model, first_module_name_list)
    # last_module_name_list = get_full_attr_name(tiny_model, last_module_name_list)
    # build expanded model
    expanded_model = build_expanded_net(tiny_model, first_module_name_list=first_module_name_list, last_module_name_list=last_module_name_list, gpd_ratio=gpd_ratio, exclude_module_name_list=exclude_module_name_list, test_input=test_input)
    expanded_model = set_state4orepa(tiny_model, expanded_model)

    # for k, v in expanded_model.state_dict().items():
    #     try:
    #         nn.init.sparse_(v)
    #     except:
    #         pass

    # print(expanded_model.cls_attn_blocks[1].gamma1.shape)
    # assert False

    # expand and init weights
    expanded_model = expand_init_wide_model(tiny_model, expanded_model, load_pretrained=(test_input is None))

    # convert to dynamic
    dynamic_expanded_model = static2dynamic(tiny_model, expanded_model, exclude_module_name_list=exclude_module_name_list)
    # dynamic_expanded_model = expanded_model

    if test_input is not None:
        dynamic_expanded_model.eval()
        input_scale_factor = 1000
        if isinstance(test_input, list):
            for i in range(len(test_input)):
                test_input[i] *= input_scale_factor
        else:
            test_input *= input_scale_factor
        tiny_output = forward_tiny(dynamic_expanded_model, test_input)
        expanded_output = forward_expanded(dynamic_expanded_model, test_input)
        # print(dynamic_expanded_model.model[7][0].weight)

        if isinstance(tiny_output, tuple):
            _, tiny_output = tiny_output
            _, expanded_output = expanded_output
        before_diff_output = expanded_output - tiny_output

        # init tiny model with expanded model
        expanded_model = expand_init_wide_model(tiny_model, expanded_model)
        dynamic_expanded_model = static2dynamic(tiny_model, expanded_model, exclude_module_name_list=exclude_module_name_list)
        dynamic_expanded_model.eval()
        tiny_output = forward_tiny(dynamic_expanded_model, test_input)
        expanded_output = forward_expanded(dynamic_expanded_model, test_input)
        if isinstance(tiny_output, tuple):
            _, tiny_output = tiny_output
            _, expanded_output = expanded_output
        after_diff_output = expanded_output - tiny_output
        print(f'Correctness Check: The difference of predictions between tiny model and expanded model is reduced from {before_diff_output.abs().mean().data:.6f} to {after_diff_output.abs().mean().data:.6f} via inverse reparamerization')
        dynamic_expanded_model.train()
    else:
        warnings.warn("The correctness cannot be guaranteed without testing!")

    return dynamic_expanded_model


def forward_tiny(model, input_list):
    gpd_ratio = get_gpd_ratio(model)
    set_active_with_ratio_rescale(model, gpd_ratio)
    network_output = None
    if isinstance(input_list, list):
        network_output = model(*input_list)
    elif isinstance(input_list, dict):
        network_output = model(**input_list)
    else:
        network_output = model(input_list)
    # set_active_with_ratio_rescale(model, 1)
    return network_output


def forward_expanded(model, input_list):
    set_active_with_ratio_rescale(model, 1)
    network_output = None
    if isinstance(input_list, list):
        network_output = model(*input_list)
    elif isinstance(input_list, dict):
        network_output = model(**input_list)
    else:
        network_output = model(input_list)
    return network_output

def cross_entropy_loss_with_soft_target(pred, soft_target):
    # logprobs = F.log_softmax(pred, dim=-1)
    # nll_loss = -logprobs.gather(dim=-1, index=soft_target.unsqueeze(1))
    # nll_loss = nll_loss.squeeze(1)
    # return torch.mean(nll_loss)
    return torch.mean(torch.sum(-soft_target * F.log_softmax(pred, dim=-1), 1))

def LabelSmoothingCrossEntropy(x, target):
    smoothing = 0.1
    confidence = 1. - smoothing
    logprobs = F.log_softmax(x, dim=-1)
    nll_loss = torch.sum(-target * logprobs, 1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()


def enhance_with_es(model, input_list, output_expanded, loss_expanded, compute_loss_function, loss_param_list, kd_loss_type='l1', kd_loss_weight=1, warmup_es_epoch=0, current_epoch=0, tiny_loss_weight=1, expanded_loss_weight=1, skip_expanded=False, avoid_bad_teacher=False, split_grad=False, grad_mask_groups=None, optimizer=None, autocast=False, loss_scaler=None):
    # first check whether all gradients are clean
    check_clean_grad(model)

    # backward for expanded model
    if not skip_expanded:
        if loss_scaler is not None:
            loss_scaler._scaler.scale(loss_expanded).backward(create_graph=False)
        else:
            loss_expanded.backward()

    # forward using tiny model
    if autocast:
        with torch.cuda.amp.autocast(enabled=False):
            output_tiny = forward_tiny(model, input_list)
    else:
        output_tiny = forward_tiny(model, input_list)


    if isinstance(loss_param_list, list):
        loss_tiny = compute_loss_function(output_tiny, *loss_param_list)
    else:
        loss_tiny = compute_loss_function(output_tiny, loss_param_list)

    orig_output_tiny = output_tiny
    if isinstance(orig_output_tiny, tuple):
        _, output_tiny = orig_output_tiny
    else:
        output_tiny = orig_output_tiny

    if isinstance(output_expanded, tuple):
        _, output_expanded = output_expanded
    # detach the prediction of expanded model
    soft_logits = output_expanded.detach()

    # if the output of compute_loss_function is a tuple, we always take the first element as the loss value
    if isinstance(loss_tiny, tuple):
        loss_tiny, *losses = loss_tiny

    # distill loss between tiny and expanded model
    if kd_loss_type == 'mse':
        kd_loss = F.mse_loss(output_tiny, soft_logits)
    elif kd_loss_type == 'l1':
        kd_loss = F.l1_loss(output_tiny, soft_logits)
    elif kd_loss_type == 'bce':
        kd_loss = F.binary_cross_entropy_with_logits(output_tiny, soft_logits)
    elif kd_loss_type == 'ce':
        kd_loss = cross_entropy_loss_with_soft_target(output_tiny, F.softmax(soft_logits, dim=1))
    elif kd_loss_type == 'sce':
        kd_loss = LabelSmoothingCrossEntropy(output_tiny, F.softmax(soft_logits, dim=1))
    else:
        assert False, f"unsupported loss type {kd_loss_type}"

    kd_loss = kd_loss_weight * kd_loss
    total_loss = loss_tiny + kd_loss

    # loss_expanded_ttt, *_ = compute_loss_function(output_expanded_origin, *loss_param_list)
    # print(loss_tiny.item(), kd_loss.item(), loss_expanded_ttt.item(), loss_expanded.item(), total_loss.item(), tiny_loss_weight)

    return total_loss, kd_loss, orig_output_tiny


def enhance_with_es_ksd(model, input_list, output_expanded, loss_expanded, compute_loss_function, loss_param_list, kd_loss_type='l1', kd_loss_weight=1):
    # first check whether all gradients are clean
    check_clean_grad(model)
    # detach the prediction of expanded model
    if isinstance(output_expanded, tuple):
        output_feat, output_pred = output_expanded
        if isinstance(output_feat, torch.Tensor):
            output_expanded = output_feat
        if isinstance(output_pred, torch.Tensor):
            output_expanded = output_pred

    soft_logits = output_expanded.detach()
    # backward for expanded model
    # if not accumulate_loss_wo_backward:
    #     loss_expanded.backward()

    # forward using tiny model
    output_tiny = forward_tiny(model, input_list)
    if isinstance(loss_param_list, list):
        loss_tiny = compute_loss_function(output_tiny, *loss_param_list)
    else:
        loss_tiny = compute_loss_function(output_tiny, loss_param_list)

    # if the output of compute_loss_function is a tuple, we always take the first element as the loss value
    if isinstance(loss_tiny, tuple):
        loss_tiny = loss_tiny[0]

    orig_output_tiny = output_tiny
    if isinstance(orig_output_tiny, tuple):
        output_feat, output_pred = orig_output_tiny
        if isinstance(output_feat, torch.Tensor):
            output_tiny = output_feat
        if isinstance(output_pred, torch.Tensor):
            output_tiny = output_pred
    else:
        output_tiny = orig_output_tiny


    # distill loss between tiny and expanded model
    if kd_loss_type == 'mse':
        kd_loss = F.mse_loss(output_tiny, soft_logits)
    elif kd_loss_type == 'l1':
        kd_loss = F.l1_loss(output_tiny, soft_logits)
    elif kd_loss_type == 'bce':
        kd_loss = F.binary_cross_entropy_with_logits(output_tiny, soft_logits)
    elif kd_loss_type == 'ce':
        kd_loss = cross_entropy_loss_with_soft_target(output_tiny, F.softmax(soft_logits, dim=1))
    elif kd_loss_type == 'sce':
        kd_loss = LabelSmoothingCrossEntropy(output_tiny, F.softmax(soft_logits, dim=1))
    else:
        assert False, f"unsupported loss type {kd_loss_type}"

    if isinstance(loss_tiny, dict) and isinstance(loss_expanded, dict):
        for k, v in loss_tiny.items():
            loss_expanded[k+'_tiny'] = v

        loss_expanded['gpd'] = kd_loss_weight * kd_loss

    return loss_expanded, orig_output_tiny


def check_clean_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad.zero_()
        # if param.grad is None or param.grad.mean().abs().data > 0:
        #     pass
        # else:
        #     warnings.warn(f'The gradient of param {name} is not clean')


def correct_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad = torch.nan_to_num(param.grad)


def print_module_names(model):
    for name, module in model.named_modules():
        print(name)
        print(module)


def get_full_attr_name(model, module_name_list):
    assert isinstance(module_name_list, list), f'last_module_name_list: {module_name_list} is not a list'
    candidate_list = []
    for i in range(len(module_name_list)):
        candidate_list.append([])
    for name, param in model.named_parameters():
        for i in range(len(module_name_list)):
            if module_name_list[i] in name:
                name = name.split(module_name_list[i])[0] + module_name_list[i]
                candidate_list[i].append(name)
    full_name_candidate_list = []
    for i in range(len(module_name_list)):
        assert len(candidate_list[i]) > 0, f'The {i+1}-th element in last_module_name_list cannot be found in the model'
        assert recursive_hasattr(model, candidate_list[i][-1])
        full_name_candidate_list.append(candidate_list[i][-1])
    return full_name_candidate_list


def detect_abnormal_param(model):
    normal_param_name_list = ['weight', 'bias']
    nnparam_name_list = []
    for name in model.state_dict().keys():
    #     parameter = recursive_getattr(model, name)
        is_normal_param = False
        for normal_param_name in normal_param_name_list:
            if name.endswith(normal_param_name):
                is_normal_param = True

        if not is_normal_param:
            nnparam_name_list.append(name)

def extract_tiny_net(model, test_sample):
    model.eval()
    gpd_ratio = get_gpd_ratio(model)
    set_active_with_ratio_rescale(model, gpd_ratio)
    network_output = None
    if isinstance(test_sample, list):
        network_output = model(*test_sample)
    else:
        network_output = model(test_sample)
    tiny_model = dynamic2static(model)
    return tiny_model


def cagrad(grads, alpha=0.5, rescale=0):
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = alpha * g0_norm
    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-8) + 0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-8)
    lmbda = coef / (gw_norm+1e-8)
    g = (0.5+lmbda*x) * g1 + (0.5+lmbda*(1-x)) * g2 # g0 + lmbda*gw
    if rescale== 0:
        return g
    elif rescale== 1:
        return g / (1+alpha**2)
    else:
        return g / (1 + alpha)

def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for p in m.parameters():
        grad = p.grad
        if grad is not None:
            grad_cur = grad.data.detach().clone()
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
        cnt += 1

def overwrite_grad(m, newgrad, grad_dims):
    newgrad = newgrad * 2 # to match the sum loss
    cnt = 0
    for param in m.parameters():
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = newgrad[beg: en].contiguous().view(param.data.size())
        param.grad = this_grad.data.clone()
        cnt += 1


def graddrop(grads):
    P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
    U = torch.rand_like(grads[:,0])
    M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
    g = (grads * M.float()).mean(1)
    return g


def copy_grad(optimizer):
    grad_groups = []
    for i in range(len(optimizer.param_groups)):
        grad = []
        for tmp_tensor in optimizer.param_groups[i]['params']:
            grad.append(tmp_tensor.grad.clone())
        grad_groups.append(grad)
    return grad_groups


def build_grad_mask(optimizer):
    grad_mask_groups = []
    contain_zero_grad = False
    mask_min = 1
    for i in range(len(optimizer.param_groups)):
        grad_mask = []
        for tmp_tensor in optimizer.param_groups[i]['params']:
            mask = (tmp_tensor.grad != 0).clone()
            mask = mask.float().expand_as(tmp_tensor.grad)
            tmp_mask_min = torch.min(mask)
            mask_min = min(mask_min, tmp_mask_min.item())
            # assert mask_min > 0, mask
            grad_mask.append(mask)
            zero_grad_count = torch.count_nonzero(1 - mask)
            if zero_grad_count > 0:
                contain_zero_grad = True
        grad_mask_groups.append(grad_mask)
    assert contain_zero_grad, 'grad does not contain zero elements'
    return grad_mask_groups


def extract_tiny_grad(optimizer, grad_mask_groups, last_grad_groups):
    grad_groups = []
    for i in range(len(optimizer.param_groups)):
        grad = []
        last_grad = last_grad_groups[i]
        grad_mask = grad_mask_groups[i]
        current_grad = optimizer.param_groups[i]['params']
        for j in range(len(current_grad)):
            grad_diff = (current_grad[j].grad - last_grad[j]).clone()
            grad.append(grad_mask[j] * grad_diff)
        grad_groups.append(grad)
    return grad_groups


def apply_grad_mask(optimizer, grad_mask_groups, tiny_grad_groups, expanded_grad_groups):
    for i in range(len(optimizer.param_groups)):
        grad_mask = grad_mask_groups[i]
        grad = tiny_grad_groups[i]
        expanded_grad = expanded_grad_groups[i]
        current_grad = optimizer.param_groups[i]['params']
        for j in range(len(current_grad)):
            merged_grad = grad_mask[j] * grad[j] + (1 - grad_mask[j]) * expanded_grad[j]
            current_grad[j].grad.data.copy_(merged_grad)

