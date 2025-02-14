# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import os
import math
import sys
from typing import Iterable, Optional
import copy

import torch
import torchvision
from torchvision import datasets, transforms
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import kornia as K

from losses import DistillationLoss
import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import build_dataset
from torchvision.utils import save_image
from timm.models.layers import trunc_normal_, DropPath
# from autoattack import AutoAttack
from gpd_utils import gpd_init, forward_tiny, forward_expanded, enhance_with_es, dynamic2static, build_grad_mask, apply_grad_mask, extract_tiny_grad
from orepa_ft import transfer2orepa
from reviewkd_utils import build_review_kd, hcl


CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

class Robust_KL_Loss(nn.KLDivLoss):
    def __init__(self, size_average=None, reduce=None, reduction='none', log_target=False):
        super(Robust_KL_Loss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)

    def forward(self, input, target):
        batch_size = input.size(0)
        loss = F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)
        loss = (1.0 / batch_size) * torch.sum(torch.sum(loss, dim=1))
        return loss


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def TRADESAttack(x, model, attack_iters, criterion_kl, step_size, epsilon):
    B, _, _, _ = x.shape
    model.eval()
    x_adv = x.detach() + 0.  # the + 0. is for copying the tensor
    x_adv += 0.001 * torch.randn(x.shape).cuda().detach()
    for _ in range(attack_iters):
        x_adv.requires_grad_()
        with torch.enable_grad():
            outputs = model(torch.cat([x, x_adv]))
            loss_kl = criterion_kl(F.log_softmax(outputs[B:], dim=1),
                                   F.softmax(outputs[0:B], dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon),
                          x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def PGDAttack(x, y, model, attack_epsilon, attack_alpha, lower_limit, loss_fn, upper_limit, max_iters, random_init):
    model.eval()

    delta = torch.zeros_like(x).cuda()
    if random_init:
        for iiiii in range(len(attack_epsilon)):
            delta[:, iiiii, :, :].uniform_(-attack_epsilon[iiiii][0][0].item(), attack_epsilon[iiiii][0][0].item())

    adv_imgs = clamp(x+delta, lower_limit, upper_limit)
    max_iters = int(max_iters)
    adv_imgs.requires_grad = True

    with torch.enable_grad():
        for _iter in range(max_iters):

            outputs = model(adv_imgs)

            loss = loss_fn(outputs, y)

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None,
                                        only_inputs=True)[0]

            adv_imgs.data += attack_alpha * torch.sign(grads.data)

            adv_imgs = clamp(adv_imgs, x-attack_epsilon, x+attack_epsilon)

            adv_imgs = clamp(adv_imgs, lower_limit, upper_limit)

    return adv_imgs.detach()


def FGSMAttack_POSTNORM(x, y, model, attack_epsilon, attack_alpha, lower_limit, loss_fn, upper_limit, args):
    orig_training_mode = model.training
    # model.eval()

    delta = torch.zeros_like(x).cuda()
    for iiiii in range(len(attack_epsilon)):
        delta[:, iiiii, :, :].uniform_(-attack_epsilon[iiiii][0][0].item(), attack_epsilon[iiiii][0][0].item())
    delta += 1e-5

    adv_imgs = clamp(x+delta, lower_limit, upper_limit)
    adv_imgs.requires_grad = True

    with torch.enable_grad():
        outputs = model(utils.normalize_data(adv_imgs, args))
        try:
            loss = loss_fn(adv_imgs, outputs, y)
        except:
            loss = loss_fn(outputs, y)
        grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None,
                                    only_inputs=True)[0]
        adv_imgs.data += attack_alpha * torch.sign(grads.data)
        adv_imgs = clamp(adv_imgs, x-attack_epsilon, x+attack_epsilon)
        adv_imgs = clamp(adv_imgs, lower_limit, upper_limit)
    model.train(orig_training_mode)
    return adv_imgs.detach()



def FGSMAttack(x, y, model, attack_epsilon, attack_alpha, lower_limit, loss_fn, upper_limit):
    orig_training_mode = model.training
    # model.eval()

    delta = torch.zeros_like(x).cuda()
    for iiiii in range(len(attack_epsilon)):
        delta[:, iiiii, :, :].uniform_(-attack_epsilon[iiiii][0][0].item(), attack_epsilon[iiiii][0][0].item())
    delta += 1e-5

    adv_imgs = clamp(x+delta, lower_limit, upper_limit)
    adv_imgs.requires_grad = True

    with torch.enable_grad():
        outputs = model(adv_imgs)
        try:
            loss = loss_fn(adv_imgs, outputs, y)
        except:
            loss = loss_fn(outputs, y)
        grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None,
                                    only_inputs=True)[0]
        adv_imgs.data += attack_alpha * torch.sign(grads.data)
        adv_imgs = clamp(adv_imgs, x-attack_epsilon, x+attack_epsilon)
        adv_imgs = clamp(adv_imgs, lower_limit, upper_limit)
    model.train(orig_training_mode)
    return adv_imgs.detach()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd_ttt_imnet(args, model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    model.eval()
    X = utils.denormalize_data(X, args)
    upper_limit, lower_limit = 1,0
    max_loss = torch.zeros(y.shape[0]).to(X.device)
    max_delta = torch.zeros_like(X).to(X.device)

    for _ in range(restarts):
        delta = torch.zeros_like(X).to(X.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            _, output = model(utils.normalize_data(X, args), utils.normalize_data(X + delta, args))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)

            extra_loss = utils.AttnDistractLoss(model)
            loss += args.extra_weight * extra_loss
            print(extra_loss)

            # print(extra_loss.item(), utils.NegAttnLoss(model).item())
            loss.backward()
            grad = delta.grad.detach()
            delta.grad = grad
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(utils.normalize_data(X+delta, args)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    perturbed_X = torch.clamp(X + max_delta, min=0, max=1)
    perturbed_X = utils.normalize_data(perturbed_X, args)
    return perturbed_X


def attack_pgd_ttt(args, model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    model.eval()
    upper_limit, lower_limit = 1,0
    max_loss = torch.zeros(y.shape[0]).to(X.device)
    max_delta = torch.zeros_like(X).to(X.device)

    for _ in range(restarts):
        delta = torch.zeros_like(X).to(X.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        for _ in range(attack_iters):
            _, output = model(utils.normalize_data(X, args), utils.normalize_data(X + delta, args))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)

            extra_loss = utils.AttnDistractLoss(model)
            loss += args.extra_weight * extra_loss

            # print(extra_loss.item(), utils.NegAttnLoss(model).item())
            loss.backward()
            grad = delta.grad.detach()
            delta.grad = grad
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(utils.normalize_data(X+delta, args)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_patch_fool_attn(args, model, x, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    model.eval()

    num_patch = 1
    patch_num_per_line = 14
    patch_size = 16
    mild_l_inf = 8 / 255.
    attack_learning_rate = 0.22
    step_size = 10
    gamma = 0.95
    train_attack_iters = 80
    atten_loss_weight = 0.002

    B, _, h, w = x.shape

    # if h != 224:
    #     X = F.upsample(x, (224,224), mode='bicubic', align_corners=False)
    # else:
    #     X = x
    delta = torch.zeros_like(x).to(x.device)
    delta.requires_grad = True
    model.zero_grad()
    _ = model(utils.normalize_data(x + delta, args))
    atten = utils.get_attn(model)

    '''choose patch'''
    atten_layer = atten[4].mean(dim=1)
    atten_layer = atten_layer.mean(dim=-2)
    max_patch_index = atten_layer.argsort(descending=True)[:, :num_patch]

    '''build mask'''
    mask = torch.zeros([x.size(0), 1, 224, 224]).to(x.device)
    for j in range(x.size(0)):
        index_list = max_patch_index[j]
        for index in index_list:
            row = (index // patch_num_per_line) * patch_size
            column = (index % patch_num_per_line) * patch_size
            mask[j, :, row:row + patch_size, column:column + patch_size] = 1
    if h != 224:
        mask = F.upsample(mask, (h, w), mode='bicubic', align_corners=False)

    '''adv attack'''
    max_patch_index_matrix = max_patch_index[:, 0]
    max_patch_index_matrix = max_patch_index_matrix.repeat(196, 1)
    max_patch_index_matrix = max_patch_index_matrix.permute(1, 0)
    max_patch_index_matrix = max_patch_index_matrix.flatten().long()

    '''constrain delta: range [x-epsilon, x+epsilon]'''
    epsilon = mild_l_inf
    delta = 2 * epsilon * torch.rand_like(x) - epsilon + x
    delta = torch.clamp(delta, 0, 1)
    X = torch.mul(x, 1 - mask)
    delta = delta.to(X.device)
    delta.requires_grad = True
    opt = torch.optim.Adam([delta], lr=attack_learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    '''Start Adv Attack'''
    for train_iter_num in range(train_attack_iters):
        model.zero_grad()
        opt.zero_grad()

        out = model(utils.normalize_data(X + torch.mul(delta, mask), args))
        atten = utils.get_attn(model)
        loss = F.cross_entropy(out, y)

        grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
        ce_loss_grad_temp = grad.view(X.size(0), -1).detach().clone()

        # Attack the first 6 layers' Attn
        range_list = range(len(atten)//2)
        for atten_num in range_list:
            if atten_num == 0:
                continue
            atten_map = atten[atten_num]
            atten_map = atten_map.mean(dim=1)
            atten_map = atten_map.view(-1, atten_map.size(-1))
            atten_map = -torch.log(atten_map)

            tmp_max_patch_index_matrix = max_patch_index_matrix
            if atten_map.size(0) != max_patch_index_matrix.size(0):
                tmp_max_patch_index_matrix = max_patch_index_matrix.reshape(B, -1)
                tmp_max_patch_index_matrix = tmp_max_patch_index_matrix.flatten()

            atten_loss = F.nll_loss(atten_map, tmp_max_patch_index_matrix)
            atten_grad = torch.autograd.grad(atten_loss, delta, retain_graph=True)[0]

            atten_grad_temp = atten_grad.view(X.size(0), -1)
            cos_sim = F.cosine_similarity(atten_grad_temp, ce_loss_grad_temp, dim=1)

            '''PCGrad'''
            atten_grad = utils.PCGrad(atten_grad_temp, ce_loss_grad_temp, cos_sim, grad.shape)
            grad += atten_grad * atten_loss_weight

        opt.zero_grad()
        delta.grad = -grad
        opt.step()
        scheduler.step()

        delta.data = clamp(delta, x-epsilon, x+epsilon)
        delta.data = torch.clamp(delta, 0, 1)

    perturb_x = X + torch.mul(delta, mask)
    max_delta = perturb_x - x
    # torch.set_printoptions(threshold=10_000)
    return max_delta


def attack_pgd_distract(args, model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    model.eval()
    upper_limit, lower_limit = 1,0
    max_loss = torch.zeros(y.shape[0]).to(X.device)
    max_delta = torch.zeros_like(X).to(X.device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(X.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            _, output = model(utils.normalize_data(X, args), utils.normalize_data(X + delta, args))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
                extra_loss = utils.AttnDistractLoss(model)
                loss += args.extra_weight * extra_loss
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(utils.normalize_data(X+delta, args)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd(args, model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, mixup=False, y_a=None, y_b=None, lam=None):
    model.eval()
    upper_limit, lower_limit = 1,0
    max_loss = torch.zeros(y.shape[0]).to(X.device)
    max_delta = torch.zeros_like(X).to(X.device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(X.device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(utils.normalize_data(X + delta, args))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(utils.normalize_data(X+delta, args)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(utils.normalize_data(X+delta, args)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def cifar_attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
                     norm, early_stop=False,
                     mixup=False, y_a=None, y_b=None, lam=None, lower_limit=0, upper_limit=1):

    model.eval()

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()

    with torch.enable_grad():
        for _ in range(restarts):
            delta = torch.zeros_like(X).cuda()
            if norm == "l_inf":
                # delta.uniform_(-epsilon, epsilon)
                for iiiii in range(len(epsilon)):
                    delta[:, iiiii, :, :].uniform_(-epsilon[iiiii][0][0].item(), epsilon[iiiii][0][0].item())
            elif norm == "l_2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*epsilon
            else:
                raise ValueError
            delta = clamp(delta, lower_limit-X, upper_limit-X)
            X.requires_grad = True
            delta.requires_grad = True
            for _ in range(attack_iters):
                output = model(X + delta)
                if early_stop:
                    index = torch.where(output.max(1)[1] == y)[0]
                else:
                    index = slice(None,None,None)
                if not isinstance(index, slice) and len(index) == 0:
                    break
                if mixup:
                    criterion = nn.CrossEntropyLoss()
                    loss = mixup_criterion(criterion, model(X+delta), y_a, y_b, lam)
                else:
                    loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                d = delta[index, :, :, :]
                g = grad[index, :, :, :]
                x = X[index, :, :, :]
                if norm == "l_inf":
                    d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
                elif norm == "l_2":
                    g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                    scaled_g = g/(g_norm + 1e-10)
                    d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
                d = clamp(d, lower_limit - x, upper_limit - x)
                delta.data[index, :, :, :] = d
                delta.grad.zero_()
            if mixup:
                criterion = nn.CrossEntropyLoss(reduction='none')
                all_loss = mixup_criterion(criterion, model(X+delta), y_a, y_b, lam)
            else:
                all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)

    delta = max_delta.detach()
    x_adv = torch.clamp(x + delta[:x.size(0)], min=lower_limit, max=upper_limit)

    return x_adv


def patch_level_aug(input1, patch_transform, upper_limit, lower_limit):
    bs, channle_size, H, W = input1.shape
    if H==224:
        ps = 16
    elif H==32:
        ps = 4
    patches = input1.unfold(2, ps, ps).unfold(3, ps, ps).permute(0,2,3,1,4,5).contiguous().reshape(-1, channle_size,ps,ps)
    patches = patch_transform(patches)

    patches = patches.reshape(bs, -1, channle_size,ps,ps).permute(0,2,3,4,1).contiguous().reshape(bs, channle_size*ps*ps, -1)
    output_images = F.fold(patches, (H,W), ps, stride=ps)
    output_images = clamp(output_images, lower_limit, upper_limit)
    return output_images

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

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
    if len(target) > 1:
        _, new_target = torch.max(target, dim=1)
    else:
        new_target = target
    gt_mask = _get_gt_mask(logits_student, new_target)
    other_mask = _get_other_mask(logits_student, new_target)
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
        / new_target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def compute_loss(pred, criterion, target, use_reviewkd=False, review_kd_loss_weight=1.0, logits_teacher=None,
                 fteacher=None, reviewkd_active=True, use_dkd=False, dkd_active=True, dkd_alpha=1.0, dkd_beta=0.5):
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
        alpha = dkd_alpha
        beta = dkd_beta
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


def train_one_epoch(logger, args, model, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, set_training_mode=True, controller_updater=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.data_set == 'CIFAR10':
        std_imagenet = torch.tensor(CIFAR_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR_MEAN).view(3,1,1).to(device)
    elif args.data_set == 'CIFAR100':
        std_imagenet = torch.tensor(CIFAR100_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(device)
    elif 'IMNET' in args.data_set:
        std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    extra_loss = None
    aux_loss = None
    for samples, targets in metric_logger.log_every(data_loader, print_freq, logger, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        acc_targets = targets

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            _, _, H, W = samples.shape
            if H==224:
                ps = 16
            elif H==32:
                ps = 4
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(ps,ps), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
            )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast(enabled=False):
            fteacher = None
            logits_teacher = None

            # if args.use_patch_aug:
            #     outputs2 = model(aug_samples)
            #     loss = criterion(outputs2, targets)
            #     loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)

            # forward for normal data
            outputs = forward_expanded(model, samples)
            # outputs, attns = model(samples, return_attn=True)
            # logits = outputs
            # compute loss
            # loss = criterion(outputs, targets)

            loss, reviewkd_loss, dkd_loss = compute_loss(outputs, criterion, targets)

            # train with es
            kd_loss = loss * 0
            tiny_result = outputs
            expanded_result = outputs

        loss_value = loss.item()

        acc1, acc5 = accuracy(tiny_result, acc_targets, topk=(1, 5))
        dt_acc1, dt_acc5 = accuracy(expanded_result, acc_targets, topk=(1, 5))

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(model, nn.parallel.DistributedDataParallel):
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.module.parameters(), create_graph=is_second_order)
        else:
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(dt_acc1=dt_acc1.item())
        metric_logger.update(kd_loss=kd_loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_gpd(logger, args, model, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, set_training_mode=True, teacher=None, grad_mask_groups=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.data_set == 'CIFAR10':
        std_imagenet = torch.tensor(CIFAR_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR_MEAN).view(3,1,1).to(device)
    elif args.data_set == 'CIFAR100':
        std_imagenet = torch.tensor(CIFAR100_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(device)
    elif 'IMNET' in args.data_set:
        std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)

    extra_loss = None
    aux_loss = None
    for samples, targets in metric_logger.log_every(data_loader, print_freq, logger, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        acc_targets = targets

        # optimizer.zero_grad()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.use_patch_aug:
            _, _, H, W = samples.shape
            if H==224:
                ps = 16
            elif H==32:
                ps = 4
            patch_transform = nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(ps,ps), scale=(0.85,1.0), ratio=(1.0,1.0), p=0.1),
                K.augmentation.RandomGaussianNoise(mean=0., std=0.01, p=0.1),
                K.augmentation.RandomHorizontalFlip(p=0.1)
            )
            aug_samples = patch_level_aug(samples, patch_transform, upper_limit, lower_limit)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        with torch.cuda.amp.autocast(enabled=False):
            if args.teacher is not None:
                with torch.no_grad():
                    fteacher, logits_teacher = teacher(samples, is_feat=True)
                    # logits_teacher = teacher(samples)
                    # fteacher = None
            else:
                fteacher = None
                logits_teacher = None

            # if args.use_patch_aug:
            #     outputs2 = model(aug_samples)
            #     loss = criterion(outputs2, targets)
            #     loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)

            # forward for normal data
            outputs = forward_expanded(model, samples)

        loss, reviewkd_loss, dkd_loss = compute_loss(outputs, criterion, targets, use_reviewkd=args.use_reviewkd, review_kd_loss_weight=args.review_kd_loss_weight, logits_teacher=logits_teacher, fteacher=fteacher, reviewkd_active=not args.deactive_kd_static_teacher, use_dkd=args.use_dkd, dkd_active=not args.deactive_kd_static_teacher, dkd_alpha=args.dkd_alpha, dkd_beta=args.dkd_beta)

        # train with es
        if not args.skip_es:
            expanded_result = outputs
            expanded_loss = loss * args.expanded_loss_weight
            loss_param_list = [criterion, targets, args.use_reviewkd, args.review_kd_loss_weight, logits_teacher, fteacher, True, args.use_dkd, True, args.dkd_alpha, args.dkd_beta]

            kd_loss_weight = args.kd_ratio if epoch >= args.start_gpd_epoch else 0

            loss, kd_loss, tiny_result = enhance_with_es(model, samples, expanded_result, expanded_loss, compute_loss, loss_param_list, kd_loss_type=args.kd_loss_type, kd_loss_weight=kd_loss_weight, warmup_es_epoch=args.warmup_es_epoch, current_epoch=epoch, skip_expanded=args.skip_expanded, avoid_bad_teacher=args.avoid_bad_teacher, split_grad=args.split_grad, grad_mask_groups=grad_mask_groups, optimizer=optimizer, autocast=True, loss_scaler=loss_scaler)
        else:
            kd_loss = loss * 0
            expanded_loss = loss
            tiny_result = outputs
            expanded_result = outputs

        if args.use_reviewkd:
            _, tiny_result = tiny_result
            _, expanded_result = expanded_result

        loss_value = loss.item()

        acc1, acc5 = accuracy(tiny_result, acc_targets, topk=(1, 5))
        dt_acc1, dt_acc5 = accuracy(expanded_result, acc_targets, topk=(1, 5))

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        # loss.backward()
        # if args.split_grad:
        #     grad_tiny = extract_tiny_grad(optimizer, grad_mask_groups, grad_expanded)
        #     apply_grad_mask(optimizer, grad_mask_groups, grad_tiny, grad_expanded)
        # grad_norm = torch.nn.utils.clip_grad_norm_(parameters=model.module.parameters(), max_norm=5, norm_type=2.0)
        # optimizer.step()

        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(model, nn.parallel.DistributedDataParallel):
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.module.parameters(), create_graph=is_second_order, split_grad=args.split_grad, grad_mask_groups=grad_mask_groups)
        else:
            loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order, split_grad=args.split_grad, grad_mask_groups=grad_mask_groups)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(eloss=expanded_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(dt_acc1=dt_acc1.item())
        metric_logger.update(kd_loss=kd_loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_postnorm(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    # switch to evaluation mode
    model.eval()

    mu_imagenet = torch.tensor((0, 0, 0)).view(3,1,1).to(device)
    std_imagenet = torch.tensor((1, 1, 1)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)
    attack_epsilon = (args.epsilon / 255.) / std_imagenet
    attack_alpha = (args.pgd_alpha / 255.) / std_imagenet

    for images, target in metric_logger.log_every(data_loader, 10, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            adv_input = FGSMAttack_POSTNORM(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, args)
        elif adv == "PGD":
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (0.5 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)
        else:
            adv_input = images

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(utils.normalize_data(adv_input, args))
            else:
                output = model(utils.normalize_data(images, args))
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, indices_in_1k=None, model2=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    # switch to evaluation mode
    model.eval()
    # model.train()
    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         module.training = False


    if args.data_set == 'CIFAR10':
        std_imagenet = torch.tensor(CIFAR_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR_MEAN).view(3,1,1).to(device)
    elif args.data_set == 'CIFAR100':
        std_imagenet = torch.tensor(CIFAR100_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(device)
    elif 'IMNET' in args.data_set:
        std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)
    attack_epsilon = (args.epsilon / 255.) / std_imagenet
    attack_alpha = (args.pgd_alpha / 255.) / std_imagenet

    for images, target in metric_logger.log_every(data_loader, 10, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            adv_input = FGSMAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit)
        elif adv == "PGD":
            # if 'IMNET' in args.data_set:
            #     std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            #     mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            #     attack_epsilon = (1 / 255.) / std_imagenet
            #     attack_alpha = (0.5 / 255.) / std_imagenet
            #     # attack_epsilon = (8 / 255.) / std_imagenet
            #     # attack_alpha = (2 / 255.) / std_imagenet
            #     upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            #     lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            if model2 is not None:
                adv_input = PGDAttack(images, target, model2, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)
            else:
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)
        else:
            adv_input = images

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                if indices_in_1k is not None:
                    output = model(images)[:,indices_in_1k]
                else:
                    # if 'nt_' in args.model:
                    #     output, outputs_noise, attn, tokenaug_index_list = model(images)
                    # else:
                    #     output = model(images)
                    output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # acc1, acc5 = accuracy(outputs_noise, target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_gpd(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, indices_in_1k=None, model2=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    # switch to evaluation mode
    model.eval()

    if args.data_set == 'CIFAR10':
        std_imagenet = torch.tensor(CIFAR_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR_MEAN).view(3,1,1).to(device)
    elif args.data_set == 'CIFAR100':
        std_imagenet = torch.tensor(CIFAR100_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(device)
    elif 'IMNET' in args.data_set:
        std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)
    attack_epsilon = (args.epsilon / 255.) / std_imagenet
    attack_alpha = (args.pgd_alpha / 255.) / std_imagenet

    for images, target in metric_logger.log_every(data_loader, 10, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            adv_input = FGSMAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit)
        elif adv == "PGD":
            if model2 is not None:
                adv_input = PGDAttack(images, target, model2, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)
            else:
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)
        else:
            adv_input = images

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                if indices_in_1k is not None:
                    output = model(images)[:,indices_in_1k]
                else:
                    # output = model(images)
                    # tiny model
                    output = forward_tiny(model, images)
                    if args.use_reviewkd:
                        _, output = output
                    # expanded model
                    expanded_output = forward_expanded(model, images)
                    if args.use_reviewkd:
                        _, expanded_output = expanded_output
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        expanded_acc1, expanded_acc5 = accuracy(expanded_output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['expanded_acc1'].update(expanded_acc1.item(), n=batch_size)
        metric_logger.meters['expanded_acc5'].update(expanded_acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} DT_Acc@1 {dt_top1.global_avg:.3f} DT_Acc@5 {dt_top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, dt_top1=metric_logger.expanded_acc1, dt_top5=metric_logger.expanded_acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_inc_clean_attention_noisy_feature(logger, data_loader_clean, data_loader_noisy, model, device, mask=None, adv=None, args=None, eval_header=None, indices_in_1k=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    # switch to evaluation mode
    model.eval()

    # for i, data in metric_logger.log_every(zip(data_loader_clean, data_loader_noisy), 10, logger, header):
    for i, data in enumerate(zip(data_loader_clean, data_loader_noisy)):
        images_clean = data[0][0]
        target_clean = data[0][1]
        images = data[1][0]
        target = data[1][1]

        images_clean = images_clean.to(device, non_blocking=True)
        target_clean = target_clean.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        utils.activate_clean_attention_noisy_feature(model)
        # compute output
        with torch.cuda.amp.autocast():
            _, output = model(images_clean, images)
            if indices_in_1k is not None:
                output = output[:,indices_in_1k]
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if i%10 == 0:
            logger.info(str(metric_logger) + str(i))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_clean_attention_noisy_feature(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, model2=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    if args.data_set == 'CIFAR10':
        std_imagenet = torch.tensor(CIFAR_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR_MEAN).view(3,1,1).to(device)
    elif args.data_set == 'CIFAR100':
        std_imagenet = torch.tensor(CIFAR100_STD).view(3,1,1).to(device)
        mu_imagenet = torch.tensor(CIFAR100_MEAN).view(3,1,1).to(device)
    elif 'IMNET' in args.data_set:
        std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
        mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
    upper_limit = ((1 - mu_imagenet)/ std_imagenet)
    lower_limit = ((0 - mu_imagenet)/ std_imagenet)
    attack_epsilon = (args.epsilon / 255.) / std_imagenet
    attack_alpha = (args.pgd_alpha / 255.) / std_imagenet

    # switch to evaluation mode
    model.eval()

    mask_matrix = None
    for images, target in metric_logger.log_every(data_loader, 10, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        utils.deactivate_clean_attention_noisy_feature(model)
        # utils.deactivate_noisy_attention_clean_feature(model)
        if adv == 'FGSM':
            adv_input = FGSMAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)

                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                if model2 is not None:
                    adv_input = PGDAttack(images, target, model2, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)
                else:
                    adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                utils.activate_clean_attention_noisy_feature(model)
                # utils.activate_noisy_attention_clean_feature(model)
                _, output = model(images, adv_input)
            else:
                output = model(images)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_clean(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:' if eval_header is None else eval_header

    # switch to evaluation mode
    model.eval()

    mask_matrix = None
    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                images = utils.normalize_data(images, args)
                output = model(images)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
                .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_pgd_noisy_feature(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TestPGD:' if eval_header is None else eval_header

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # switch to evaluation mode
    model.eval()

    mask_matrix = None
    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                utils.deactivate_noisy_attention_clean_feature(model)
                delta = attack_pgd(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                delta = delta.detach()
                images_adv = utils.normalize_data(torch.clamp(images + delta[:images.size(0)], min=0, max=1), args)
                utils.activate_noisy_attention_clean_feature(model)
                images = utils.normalize_data(images, args)
                output, _ = model(images, images_adv)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.meters['pgd_acc'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.pgd_acc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_pgd_noisy_attention(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TestPGD:' if eval_header is None else eval_header

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # switch to evaluation mode
    model.eval()

    mask_matrix = None
    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                utils.deactivate_clean_attention_noisy_feature(model)
                delta = attack_pgd(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                delta = delta.detach()
                images_adv = utils.normalize_data(torch.clamp(images + delta[:images.size(0)], min=0, max=1), args)
                utils.activate_clean_attention_noisy_feature(model)
                _, output = model(images, images_adv)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.meters['pgd_acc'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.pgd_acc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_pgd_ttt(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TestPGD:' if eval_header is None else eval_header

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                delta = attack_pgd_ttt(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                delta = delta.detach()
                images_adv = utils.normalize_data(torch.clamp(images + delta[:images.size(0)], min=0, max=1), args)
                output = model(images_adv)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.meters['pgd_acc'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.pgd_acc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_pgd(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TestPGD:' if eval_header is None else eval_header

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                delta = attack_pgd(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                delta = delta.detach()
                images_adv = utils.normalize_data(torch.clamp(images + delta[:images.size(0)], min=0, max=1), args)
                output = model(images_adv)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.meters['pgd_acc'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.pgd_acc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_pgd_patchfool(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TestPGD:' if eval_header is None else eval_header

    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # switch to evaluation mode
    model.eval()
    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input)
            else:
                # delta = attack_pgd_negattn(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                delta = attack_patch_fool_attn(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                # delta = delta.detach()
                images_adv = utils.normalize_data(torch.clamp(images + delta[:images.size(0)], min=0, max=1), args)
                output = model(images_adv)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.meters['pgd_acc'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.pgd_acc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def evaluate_pgd_trades(logger, data_loader, model, device, mask=None, adv=None, args=None, eval_header=None, controller=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TestPGD:' if eval_header is None else eval_header

    epsilon = 0.031
    pgd_alpha = 0.007

    # switch to evaluation mode
    model.eval()

    mask_matrix = None
    for images, target in metric_logger.log_every(data_loader, 1, logger, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if adv == 'FGSM':
            std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
            mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            attack_epsilon = (1 / 255.) / std_imagenet
            attack_alpha = (1 / 255.) / std_imagenet
            upper_limit = ((1 - mu_imagenet)/ std_imagenet)
            lower_limit = ((0 - mu_imagenet)/ std_imagenet)
            adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=1, random_init=False)
        elif adv == "PGD":
            if args.data_set == 'CIFAR10':
                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
                CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
                mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
                std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()
                attack_epsilon = (args.epsilon / 255.) / std
                attack_alpha = (args.pgd_alpha / 255.) / std
                upper_limit = ((1 - mu)/ std)
                lower_limit = ((0 - mu)/ std)
                adv_input = cifar_attack_pgd(model, images, target, attack_epsilon, attack_alpha, args.attack_iters_test, restarts=1, norm='l_inf', lower_limit=lower_limit, upper_limit=upper_limit)

            elif 'IMNET' in args.data_set:
                std_imagenet = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
                mu_imagenet = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
                attack_epsilon = (1 / 255.) / std_imagenet
                attack_alpha = (0.5 / 255.) / std_imagenet
                upper_limit = ((1 - mu_imagenet)/ std_imagenet)
                lower_limit = ((0 - mu_imagenet)/ std_imagenet)
                adv_input = PGDAttack(images, target, model, attack_epsilon, attack_alpha, lower_limit, criterion, upper_limit, max_iters=5, random_init=True)

        # compute output
        with torch.cuda.amp.autocast():
            if adv:
                output = model(adv_input, mask_matrix=mask_matrix, mask_layer_index=args.mask_layer_index if args.mask_layer_index > 0 else None)
            else:
                delta = attack_pgd(args, model, images, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
                delta = delta.detach()
                images_adv = utils.normalize_data(torch.clamp(images + delta[:images.size(0)], min=0, max=1), args)
                output = model(images_adv, mask_matrix=mask_matrix, mask_layer_index=args.mask_layer_index if args.mask_layer_index > 0 else None)
            loss = criterion(output, target)

        if mask is None:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(output[:,mask], target, topk=(1, 5))


        batch_size = images.shape[0]
        metric_logger.meters['pgd_acc'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.pgd_acc))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_inc_ttt(logger, data_loader, model, device, args=None):
    """Evaluate network on given corrupted dataset."""

    model.eval()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    result_dict = {}
    ce_alexnet = utils.get_ce_alexnet()

    # transform for imagenet-c
    inc_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    for name, path in utils.data_loaders_names.items():
        for severity in range(1, 6):
            inc_dataset = torchvision.datasets.ImageFolder(os.path.join(args.inc_path, path, str(severity)), transform=inc_transform)

            sampler_val = torch.utils.data.DistributedSampler(
                inc_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            inc_data_loader = torch.utils.data.DataLoader(
                inc_dataset, sampler=sampler_val, batch_size=int(args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            test_stats = evaluate_inc_clean_attention_noisy_feature(logger, data_loader, inc_data_loader, model, device, args=args)
            logger.info(f"Accuracy on the {name+'({})'.format(severity)}: {test_stats['acc1']:.1f}%")
            result_dict[name+'({})'.format(severity)] = test_stats['acc1']

    mCE = 0
    counter = 0
    overall_acc = 0
    for name, path in utils.data_loaders_names.items():
        acc_top1 = 0
        for severity in range(1, 6):
            acc_top1 += result_dict[name+'({})'.format(severity)]
        acc_top1 /= 5
        CE = utils.get_mce_from_accuracy(acc_top1, ce_alexnet[name])
        mCE += CE
        overall_acc += acc_top1
        counter += 1
        logger.info("{0}: Top1 accuracy {1:.2f}, CE: {2:.2f}".format(
            name, acc_top1, 100. * CE))

    overall_acc /= counter
    mCE /= counter
    logger.info("Corruption Top1 accuracy {0:.2f}, mCE: {1:.2f}".format(overall_acc, mCE * 100.))
    return mCE * 100.


@torch.no_grad()
def eval_inc(logger, model, device, args=None, target_type=None):
    """Evaluate network on given corrupted dataset."""

    model.eval()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    result_dict = {}
    ce_alexnet = utils.get_ce_alexnet()

    # transform for imagenet-c
    inc_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(224),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    for name, path in utils.data_loaders_names.items():
        for severity in range(1, 6):
            if target_type is not None and (path != target_type or severity != 5):
                continue
            inc_dataset = torchvision.datasets.ImageFolder(os.path.join(args.inc_path, path, str(severity)), transform=inc_transform)

            sampler_val = torch.utils.data.DistributedSampler(
                inc_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

            inc_data_loader = torch.utils.data.DataLoader(
                inc_dataset, sampler=sampler_val, batch_size=int(args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            test_stats = evaluate(logger, inc_data_loader, model, device, args=args)
            logger.info(f"Accuracy on the {name+'({})'.format(severity)}: {test_stats['acc1']:.1f}%")
            result_dict[name+'({})'.format(severity)] = test_stats['acc1']
            if target_type is not None:
                return test_stats['acc1']

    mCE = 0
    counter = 0
    overall_acc = 0
    for name, path in utils.data_loaders_names.items():
        acc_top1 = 0
        for severity in range(1, 6):
            acc_top1 += result_dict[name+'({})'.format(severity)]
        acc_top1 /= 5
        CE = utils.get_mce_from_accuracy(acc_top1, ce_alexnet[name])
        mCE += CE
        overall_acc += acc_top1
        counter += 1
        logger.info("{0}: Top1 accuracy {1:.2f}, CE: {2:.2f}".format(
            name, acc_top1, 100. * CE))

    overall_acc /= counter
    mCE /= counter
    logger.info("Corruption Top1 accuracy {0:.2f}, mCE: {1:.2f}".format(overall_acc, mCE * 100.))
    return mCE * 100.


@torch.no_grad()
def eval_cifarc(logger, model, device, args=None, target_type=None):
    """Evaluate network on given corrupted dataset."""
    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]

    # switch to evaluation mode
    model.eval()

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    test_data, _ = build_dataset(False, args)
    base_path = args.cifarc_base_path

    corruption_accs = []
    for corruption in CORRUPTIONS:
        if target_type is not None and corruption != target_type:
            continue
        # Reference to original data is mutated
        test_data.data = np.load(os.path.join(base_path, corruption + '.npy'))
        test_data.targets = torch.LongTensor(np.load(os.path.join(base_path, 'labels.npy')))

        sampler_val = torch.utils.data.DistributedSampler(
            test_data, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        test_loader_val = torch.utils.data.DataLoader(
            test_data, sampler=sampler_val,
            batch_size=int(args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        test_stats = evaluate(logger, test_loader_val, model, device, args=args, eval_header=corruption)
        corruption_accs.append(test_stats['acc1'])
    logger.info(f'mCE: {np.mean(corruption_accs):.2f}')

    return np.mean(corruption_accs)




def LayerExploreLoss(attn, th=2):
    sim_sum = 0
    counter = 1e-6
    for i in range(len(attn)-1):
        mask0 = attn[i].mean(dim=1).squeeze()
        mask1 = attn[i+1].mean(dim=1).squeeze()
        n_tokens0 = mask0.shape[-1]
        n_tokens1 = mask1.shape[-1]
        if n_tokens0!=n_tokens1:
            continue
        threshold = th/n_tokens0
        score0 = torch.mean(mask0, dim=1, keepdim=True)
        score1 = torch.mean(mask1, dim=1, keepdim=True)
        score0 = (score0 > threshold) * (score0)
        score1 = (score1 > threshold) * (score1)
        # _, index0 = torch.topk(score0, 10)
        # _, index1 = torch.topk(score1, 10)
        sim = F.cosine_similarity(score0, score1, dim=-1)
        sim = sim.mean()
        sim_sum += sim
        counter += 1
    sim_sum = sim_sum / counter
    return sim_sum


def TokenExploreLoss(attn, th=2):
    sim_sum = 0
    counter = 1e-6
    for i in range(len(attn)):
        mask0 = attn[i].mean(dim=1).squeeze()
        n_tokens = mask0.shape[-1]
        threshold = th/n_tokens
        score0 = torch.mean(mask0, dim=1, keepdim=True)
        mask0 = (mask0 > threshold) * (mask0)
        score0 = (score0 > threshold) * (score0)
        sim = F.cosine_similarity(score0, mask0, dim=-1)
        sim = sim.mean()
        sim_sum += sim
        counter += 1
    sim_sum = sim_sum / counter
    return sim_sum





