# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import utils

CIFAR_MEAN = (0.49139968, 0.48215827, 0.44653124)
CIFAR_STD = (0.24703233, 0.24348505, 0.26158768)

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args, use_test=False):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if use_test:
            root = args.test_data
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET100':
        if use_test:
            root = args.test_data
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    if is_train and args.deepaugment:
        if args.data_set == 'CIFAR10' or args.data_set == 'CIFAR100':
            assert 'CIFAR' in args.deepaugment_base_path, 'invalid deepaugment_base_path: %s' % args.deepaugment_base_path
        elif args.data_set == 'IMNET':
            assert 'CIFAR' not in args.deepaugment_base_path, 'invalid deepaugment_base_path: %s' % args.deepaugment_base_path
        edsr_data = datasets.ImageFolder(os.path.join(args.deepaugment_base_path, 'EDSR'), transform)
        cae_data = datasets.ImageFolder(os.path.join(args.deepaugment_base_path, 'CAE'), transform)
        dataset = torch.utils.data.ConcatDataset([dataset, edsr_data, cae_data])


    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if args.data_set == 'CIFAR10':
        data_mean = CIFAR_MEAN
        data_std = CIFAR_STD
    elif args.data_set == 'CIFAR100':
        data_mean = CIFAR100_MEAN
        data_std = CIFAR100_STD
    elif 'IMNET' in args.data_set:
        data_mean = IMAGENET_DEFAULT_MEAN
        data_std = IMAGENET_DEFAULT_STD
    else:
        assert False, '%s not supported when creating transformations' % args.data_set
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=data_mean,
            std=data_std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(data_mean, data_std))
    return transforms.Compose(t)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]