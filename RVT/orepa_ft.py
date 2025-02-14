import torch
import torch.nn as nn
import torch.nn.init as init
import math
import numpy as np
import torch.nn.functional as F
import copy


def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


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

def transfer2orepa_single(model, attr_name, query_class, single_branch_preserve, train_from_scratch):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if len(attr_name_split) == 1:
        value = None
        source_model = getattr(model, module_name)
        if isinstance(source_model, query_class):
            new_weight = source_model.weight
            new_bias = source_model.bias
            value = OREPA(source_model.in_channels, source_model.out_channels, source_model.kernel_size[0], source_model.stride[0], source_model.padding[0], source_model.dilation[0], source_model.groups, single_init=not train_from_scratch, single_branch_preserve=single_branch_preserve)
            with torch.no_grad():
                value.weight_orepa_origin.copy_(new_weight)
                if new_bias is not None:
                    value.bias_orepa_origin.copy_(new_bias)
        # model.__delattr__(module_name)
        setattr(model, module_name, value)
        return
    transfer2orepa_single(getattr(model, module_name), attr_name[len(module_name) + 1:], query_class, single_branch_preserve, train_from_scratch)


def transfer2orepa(model, single_branch_preserve=False, return_transfered_list=False, train_from_scratch=False, exclude_module_name_list=[], print_info=True, contain_orepa=False):
    new_model = copy.deepcopy(model)
    query_class = nn.Conv2d
    transfered_module_list = []

    for name_module1 in model.named_modules():
        name1, m1 = name_module1
        # if 'loss' in name1:
        #     continue
        # if isinstance(m1, query_class) and m1.kernel_size[0]==3 and m1.padding[0] == m1.kernel_size[0] // 2 and name1 not in exclude_module_name_list and m1.groups == 1:
        if isinstance(m1, query_class) and m1.stride[0]==1 and m1.kernel_size[0]==3 and m1.padding[0] == m1.kernel_size[0] // 2 and (contain_orepa or m1.out_channels < 700) and name1 not in exclude_module_name_list:
            if print_info:
                print(f'Transfer {name1}: {m1} to OREPA block')
            transfer2orepa_single(new_model, name1, query_class, single_branch_preserve, train_from_scratch)
            transfered_module_list.append(name1)

    if return_transfered_list:
        return new_model, transfered_module_list
    else:
        return new_model


def transfer2orepa4SE_single(model, attr_name, query_class, single_branch_preserve, train_from_scratch):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if len(attr_name_split) == 1:
        value = None
        source_model = getattr(model, module_name)
        if isinstance(source_model, query_class):
            value = OREPA_SE_4x(
                source_model.res_conv1.in_channels, source_model.res_conv1.out_channels, 3, 1, 1, 1, 1, single_init=not train_from_scratch, single_branch_preserve=single_branch_preserve
            )
            # copy activate
            new_activate = copy.deepcopy(source_model.activate)
            value.__delattr__('nonlinear')
            setattr(value, 'nonlinear', new_activate)
            # copy res_conv1
            res_conv1_weight = source_model.res_conv1.weight
            res_conv1_bias = source_model.res_conv1.bias
            with torch.no_grad():
                value.weight_orepa_origin.copy_(res_conv1_weight)
                if res_conv1_bias is not None:
                    value.bias_orepa_origin.copy_(res_conv1_bias)
            # copy res_deep_conv3_stack_list
            for i in range(value._3CONV_BR_NUM):
                # res_deep_conv1_expand
                res_deep_conv1_expand_weight = source_model.res_deep_conv3_stack_list[i][0].weight
                res_deep_conv1_expand_bias = source_model.res_deep_conv3_stack_list[i][0].bias
                with torch.no_grad():
                    value.res_deep_conv3_stack_list[i][0].copy_(res_deep_conv1_expand_weight)
                    if res_deep_conv1_expand_bias is not None:
                        value.bias_res_deep_conv3_stack_list[i][0].copy_(res_deep_conv1_expand_bias)
                # res_deep_conv3
                res_deep_conv3_weight = source_model.res_deep_conv3_stack_list[i][1].weight
                res_deep_conv3_bias = source_model.res_deep_conv3_stack_list[i][1].bias
                with torch.no_grad():
                    value.res_deep_conv3_stack_list[i][1].copy_(res_deep_conv3_weight)
                    if res_deep_conv3_bias is not None:
                        value.bias_res_deep_conv3_stack_list[i][1].copy_(res_deep_conv3_bias)
                # res_deep_conv1_squeeze
                res_deep_conv1_squeeze_weight = source_model.res_deep_conv3_stack_list[i][2].weight
                res_deep_conv1_squeeze_bias = source_model.res_deep_conv3_stack_list[i][2].bias
                with torch.no_grad():
                    value.res_deep_conv3_stack_list[i][2].copy_(res_deep_conv1_squeeze_weight)
                    if res_deep_conv1_squeeze_bias is not None:
                        value.bias_res_deep_conv3_stack_list[i][2].copy_(res_deep_conv1_squeeze_bias)
            # copy scale
            new_scale = source_model.scale
            with torch.no_grad():
                value.scale.copy_(new_scale)
        setattr(model, module_name, value)
        return
    transfer2orepa4SE_single(getattr(model, module_name), attr_name[len(module_name) + 1:], query_class, single_branch_preserve, train_from_scratch)


def transfer2orepa4SE(model, query_class=nn.Conv2d, single_branch_preserve=False, return_transfered_list=False, train_from_scratch=False):
    new_model = copy.deepcopy(model)
    transfered_module_list = []
    for name_module1 in model.named_modules():
        name1, m1 = name_module1
        if isinstance(m1, query_class):
            print(f'Transfer {name1} to OREPA_SE_4x block')
            transfer2orepa4SE_single(new_model, name1, query_class, single_branch_preserve, train_from_scratch)
            transfered_module_list.append(name1)
    if return_transfered_list:
        return new_model, transfered_module_list
    else:
        return new_model


def orepa2plain_single(model, attr_name, target_class):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if len(attr_name_split) == 1:
        value = None
        source_model = getattr(model, module_name)
        source_model.switch_to_deploy()
        if isinstance(source_model, OREPA):
            source_conv = source_model.orepa_reparam
            value = target_class(
                source_conv.in_channels,
                source_conv.out_channels,
                source_conv.kernel_size[0],
                stride=source_conv.stride,
                padding=source_conv.padding,
                dilation=source_conv.dilation,
                groups=source_conv.groups,
                bias=source_conv.bias is not None,
            )
            value.load_state_dict(source_conv.state_dict())
        if isinstance(source_model, OREPA_SE_4x):
            source_conv = source_model.orepa_reparam
            value = target_class(
                source_conv.in_channels,
                source_conv.out_channels,
                prelu=True,
                is_train=False
            )
            value.merged = True
            value.res_conv3.load_state_dict(source_conv.state_dict())
            # copy activate
            new_activate = copy.deepcopy(source_model.nonlinear)
            value.__delattr__('activate')
            setattr(value, 'activate', new_activate)

        # model.__delattr__(module_name)
        setattr(model, module_name, value)
        return
    orepa2plain_single(getattr(model, module_name), attr_name[len(module_name) + 1:], target_class)


def orepa2plain(model, target_class=nn.Conv2d):
    new_model = copy.deepcopy(model)
    for name_module1 in model.named_modules():
        name1, m1 = name_module1
        if isinstance(m1, OREPA):
            orepa2plain_single(new_model, name1, target_class=nn.Conv2d)
        if isinstance(m1, OREPA_SE_4x):
            orepa2plain_single(new_model, name1, target_class=target_class)
    return new_model


def merge_conv(conv1_weight, conv1_bias, conv2_weight, conv2_bias, transposed=False):
    mid_ch = conv1_weight.size(0)
    if transposed:
        assert mid_ch == conv2_weight.size(0)
    else:
        assert mid_ch == conv2_weight.size(1)

    k1 = conv1_weight.size(3)
    k2 = conv2_weight.size(3)

    # print(f"merging conv1({in_ch}, {mid_ch}, {k1}) conv2({mid_ch}, {out_ch}, {k2})")
    assert k1 == 1 or k2 == 1
    # k = max(k1, k2)

    if transposed:
        # k2, k2, out, mid  x  k1, k1, mid, in  =  k, k, out, in
        conv_weight = torch.matmul(conv2_weight.permute(2, 3, 1, 0), conv1_weight.permute(2, 3, 0, 1))
        conv_weight = conv_weight.permute(3, 2, 0, 1)  # in, out, k, k
        if conv1_bias is not None:
            conv_bias = torch.matmul(conv2_weight.permute(1, 0, 2, 3).sum((2, 3)), conv1_bias) + conv2_bias
        else:
            conv_bias = conv2_bias
    else:
        # k2, k2, out, mid  x  k1, k1, mid, in  =  k, k, out, in
        conv_weight = torch.matmul(conv2_weight.permute(2, 3, 0, 1), conv1_weight.permute(2, 3, 0, 1))
        conv_weight = conv_weight.permute(2, 3, 0, 1)
        if conv1_bias is not None:
            conv_bias = torch.matmul(conv2_weight.sum((2, 3)), conv1_bias) + conv2_bias
        else:
            conv_bias = conv2_bias

    # print(f"\t into weight: {conv_weight.size()}, bias: {conv_bias.size()}")
    return conv_weight, conv_bias



class OREPA(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, internal_channels_1x1_3x3=None, deploy=False, nonlinear=None, single_init=False, weight_only=False, single_branch_preserve=False, init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()

        self.internal_channels_1x1_3x3 = internal_channels_1x1_3x3
        self.deploy = deploy
        self.single_init = single_init
        self.single_branch_preserve = single_branch_preserve
        self.init_hyper_para = init_hyper_para
        self.init_hyper_gamma = init_hyper_gamma

        self.weight_only = weight_only

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.branch_counter = 0

            # conv 3x3
            self.weight_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups),
                             kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.bias_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels))
            nn.init.constant_(self.bias_orepa_origin, 0.0)
            self.branch_counter += 1

            # conv1x1 + pooling
            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            # conv1x1 freq
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer(
                'weight_orepa_avg_avg',
                torch.ones(kernel_size,
                           kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 3,
                             3))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels
                # internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3,
                                int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
                # init.kaiming_uniform_(
                # self.weight_orepa_1x1_kxk_conv1, a=math.sqrt(0.0))
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels,
                             int(internal_channels_1x1_3x3 / self.groups),
                             kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size,
                             kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            if single_branch_preserve:
                self.register_buffer('vector', torch.Tensor(self.branch_counter, self.out_channels))
            else:
                self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            # if weight_only is False:
            #     self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  # origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  # avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  # prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  # 1x1_kxk
            init.constant_(self.vector[4, :], 0.05 * math.sqrt(init_hyper_gamma))  # 1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  # dws_conv

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)

            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if nonlinear is None:
                self.nonlinear = nn.Identity()
            else:
                self.nonlinear = nonlinear

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_active_init()

    def extra_repr(self):
        if 'active_in_channels' in self.__dict__:
            s = ('{active_in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, groups={groups}, dilation={dilation}')
        else:
            s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, groups={groups}, dilation={dilation}')
        return s.format(**self.__dict__)

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size,
                                    self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                                                         (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                                                         (i + 1 - half_fg) / 3)

        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                           self.weight_orepa_origin,
                                           self.vector[0, :])

        # weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_avg_avg), self.vector[1, :])

        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_prior), self.vector[2, :])

        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 +
                                          self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

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
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1,
                                                  self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                            self.vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw,
                                            self.weight_orepa_gconv_pw,
                                            self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                          self.vector[5, :])

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        bias = self.bias_orepa_origin

        return weight, bias

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)

        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i / groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))

        weight, bias = self.weight_gen()

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

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.weight_gen()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride,
                                       padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')

        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_active_init(self):
        self.init_gamma(0.0)
        # 0245
        init.constant_(self.vector[0, :], 1.0)
        # init.constant_(self.vector[2, :], 1.0)
        # init.constant_(self.vector[4, :], 1.0)
        # init.constant_(self.vector[5, :], 1.0)
        # init.constant_(self.vector[1, :], 1.0)
        # init.constant_(self.vector[3, :], 1.0)


class OREPA_SE_4x(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False,
                 nonlinear=None,
                 single_init=False,
                 weight_only=False,
                 single_branch_preserve=False,
                 init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA_SE_4x, self).__init__()

        self.internal_channels_1x1_3x3 = internal_channels_1x1_3x3
        self.deploy = deploy
        self.single_init = single_init
        self.single_branch_preserve = single_branch_preserve
        self.init_hyper_para = init_hyper_para
        self.init_hyper_gamma = init_hyper_gamma

        self._3CONV_BR_NUM = 4
        self._EXPAND = 2

        self.weight_only = weight_only

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if deploy:
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding, dilation=dilation, groups=groups, bias=True)

        else:

            self.branch_counter = 0

            # single expert 4x
            ## branch1
            self.weight_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups),
                             1, 1))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.bias_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels))
            nn.init.constant_(self.bias_orepa_origin, 0.0)
            ## branch2-5
            self.res_deep_conv3_stack_list = nn.ModuleList()
            self.bias_res_deep_conv3_stack_list = nn.ModuleList()
            for i in range(self._3CONV_BR_NUM):
                # conv1
                res_deep_conv1_expand = nn.Parameter(
                    torch.Tensor(in_channels * self._EXPAND, int(in_channels / self.groups),
                                 1, 1))
                init.kaiming_uniform_(res_deep_conv1_expand, a=math.sqrt(0.0))
                bias_res_deep_conv1_expand = nn.Parameter(
                    torch.Tensor(in_channels * self._EXPAND))
                nn.init.constant_(bias_res_deep_conv1_expand, 0.0)
                # conv2
                res_deep_conv3 = nn.Parameter(
                    torch.Tensor(in_channels * self._EXPAND, in_channels * self._EXPAND,
                                 3, 3))
                init.kaiming_uniform_(res_deep_conv3, a=math.sqrt(0.0))
                bias_res_deep_conv3 = nn.Parameter(
                    torch.Tensor(in_channels * self._EXPAND))
                nn.init.constant_(bias_res_deep_conv3, 0.0)
                # conv3
                res_deep_conv1_squeeze = nn.Parameter(
                    torch.Tensor(out_channels, in_channels * self._EXPAND,
                                 1, 1))
                init.kaiming_uniform_(res_deep_conv1_squeeze, a=math.sqrt(0.0))
                bias_res_deep_conv1_squeeze = nn.Parameter(
                    torch.Tensor(out_channels))
                nn.init.constant_(bias_res_deep_conv1_squeeze, 0.0)

                self.res_deep_conv3_stack_list.append(
                    nn.ParameterList([res_deep_conv1_expand, res_deep_conv3, res_deep_conv1_squeeze]))
                self.bias_res_deep_conv3_stack_list.append(
                    nn.ParameterList([bias_res_deep_conv1_expand, bias_res_deep_conv3, bias_res_deep_conv1_squeeze]))
            self.scale = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))  # SE (1, 64, 1, 1)
            nn.init.constant_(self.scale.data, 1)
            self.branch_counter += 1

            # conv1x1 + pooling
            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            # conv1x1 freq
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer(
                'weight_orepa_avg_avg',
                torch.ones(kernel_size,
                           kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1,
                             1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3,
                                int(in_channels / self.groups), 1, 1))
                id_value = np.zeros(
                    (internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(
                    self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
                # init.kaiming_uniform_(
                # self.weight_orepa_1x1_kxk_conv1, a=math.sqrt(0.0))
            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels,
                             int(internal_channels_1x1_3x3 / self.groups),
                             kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size,
                             kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            if single_branch_preserve:
                self.register_buffer('vector', torch.Tensor(self.branch_counter, self.out_channels))
            else:
                self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            # if weight_only is False:
            #     self.bn = nn.BatchNorm2d(self.out_channels)

            self.fre_init()

            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  # origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  # avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  # prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  # 1x1_kxk
            init.constant_(self.vector[4, :], 0.05 * math.sqrt(init_hyper_gamma))  # 1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  # dws_conv

            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)

            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if nonlinear is None:
                self.nonlinear = nn.Identity()
            else:
                self.nonlinear = nonlinear

            if single_init:
                #   Initialize the vector.weight of origin as 1 and others as 0. This is not the default setting.
                self.single_active_init()

    def extra_repr(self):
        if 'active_in_channels' in self.__dict__:
            s = ('{active_in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, groups={groups}, dilation={dilation}')
        else:
            s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, groups={groups}, dilation={dilation}')
        return s.format(**self.__dict__)

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size,
                                    self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                                                         (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                                                         (i + 1 - half_fg) / 3)

        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        # single expert merge conv
        # 1x1 branch
        conv_1x1_weight = self.weight_orepa_origin
        conv_1x1_bias = self.bias_orepa_origin
        covnv_pad_3x3 = torch.nn.functional.pad(conv_1x1_weight, [1, 1, 1, 1])  # 单独的conv分支，可以理解为conv2
        conv_weight = covnv_pad_3x3
        conv_bias = conv_1x1_bias

        for res_deep_conv3_stack, bias_res_deep_conv3_stack in zip(self.res_deep_conv3_stack_list, self.bias_res_deep_conv3_stack_list):
            conv_deep_weight, conv_deep_bias = merge_conv(
                res_deep_conv3_stack[0],
                bias_res_deep_conv3_stack[0],
                res_deep_conv3_stack[1],
                bias_res_deep_conv3_stack[1],
            )
            conv_deep_weight, conv_deep_bias = merge_conv(
                conv_deep_weight,
                conv_deep_bias,
                res_deep_conv3_stack[2],
                bias_res_deep_conv3_stack[2],
            )
            conv_weight = conv_weight + conv_deep_weight
            conv_bias = conv_bias + conv_deep_bias

        _added = torch.zeros_like(conv_weight)
        nn.init.dirac_(_added)
        conv_weight = conv_weight * self.scale.permute(1, 0, 2, 3) + _added
        conv_bias = conv_bias * self.scale.reshape(-1)
        weight_orepa_origin = conv_weight
        weight_orepa_origin = torch.einsum('oihw,o->oihw',
                                           weight_orepa_origin,
                                           self.vector[0, :])

        # weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_avg_avg), self.vector[1, :])

        weight_orepa_pfir = torch.einsum(
            'oihw,o->oihw',
            torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                         self.weight_orepa_prior), self.vector[2, :])

        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 +
                                          self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

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
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1,
                                                  self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1,
                                            self.vector[4, :])

        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw,
                                            self.weight_orepa_gconv_pw,
                                            self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv,
                                          self.vector[5, :])

        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv
        bias = conv_bias

        return weight, bias

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)

        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i / groups_conv), h, w)

    def forward(self, inputs=None):
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))

        weight, bias = self.weight_gen()

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

    def get_equivalent_kernel_bias(self):
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        if hasattr(self, 'or1x1_reparam'):
            return
        kernel, bias = self.weight_gen()
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride,
                                       padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')

        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        init.constant_(self.vector, gamma_value)

    def single_active_init(self):
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)



def rearrange_module_single(tiny_model, model, attr_name):
    assert recursive_hasattr(model, attr_name), f'model does not contain the attr_name \"{attr_name}\"'
    attr_name_split = attr_name.split(".")
    module_name = attr_name_split[0]
    if module_name == '':
        return
    if len(attr_name_split) == 1:
        source_model = getattr(model, module_name)
        setattr(model, module_name, source_model)
        return
    rearrange_module_single(getattr(tiny_model, module_name), getattr(model, module_name), attr_name[len(module_name) + 1:])


def rearrange_module(tiny_model, model):
    for name0, m0 in tiny_model.named_modules():
        # if isinstance(m0, nn.Sequential):
        rearrange_module_single(tiny_model, model, name0)
    return model