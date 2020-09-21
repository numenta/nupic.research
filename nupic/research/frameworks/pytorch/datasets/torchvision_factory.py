# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
import sys
import warnings

import torch
from filelock import FileLock
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASETS_STATS = {
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "TinyImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR10": ((0.4914009 , 0.48215914, 0.44653103),
                (0.24703224, 0.24348514, 0.26158786)),
    "CIFAR100": ((0.50707573, 0.48655039, 0.44091937),
                 (0.26733428, 0.25643846, 0.27615047)),
    "MNIST": ((0.13062755,), (0.30810780,)),
    "Omniglot_bg": ((0.92206019,), (0.26807660,)),
    "Omniglot_eval": ((0.91571581,), (0.27781320,))
}


def torchvisiondataset(root, dataset_name, train=True, download=True,
                       transform=None, target_transform=None,):
    """
    Create train and val datsets from torchvision of `dataset_name`.
    Returns None for test set.
    """

    root = os.path.expanduser(root)
    dataset_class = getattr(datasets, dataset_name)
    if transform is None:
        if dataset_name in DATASETS_STATS.keys():
            transform = base_transform(*DATASETS_STATS[dataset_name])
        else:
            try:
                dataset_stats = calculate_statistics(dataset_class, root)
                transform = base_transform(*dataset_stats)
            except Exception:
                warnings.warn(
                    f"Can't calculate statistics for {dataset_name}. "
                    "Update DATASETS_STATS manually before proceeding"
                )
                raise

    with FileLock(f"{root}.lock"):
        dataset = dataset_class(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    return dataset


def omniglot(root, train=True, download=True, evaluation=False,
             transform=None, target_transform=None):
    """
    Omniglot Torchvision dataset.
    Each classes contains 20 samples.
    Omniglot contains characters from 50 different languages

    train (bool, optional):
        If True, creates dataset from the "background" set, otherwise
        creates from the "evaluation" set.
        Background contains 964 classes, and evaluation 659 classes, all unique
    """

    if not evaluation:
        transform = base_transform_with_resize(*DATASETS_STATS["Omniglot_bg"])
    else:
        transform = base_transform_with_resize(*DATASETS_STATS["Omniglot_eval"])

    dataset_class = datasets.Omniglot
    root = os.path.expanduser(root)
    with FileLock(f"{root}.lock"):
        dataset = dataset_class(
            root=root,
            background=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    return dataset


def base_transform(stats_mean, stats_std):
    """Convert to tensor and normalize"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(stats_mean, stats_std),
    ])


def base_transform_with_resize(stats_mean, stats_std):
    """Convert to tensor and normalize"""
    return transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(stats_mean, stats_std),
    ])


def calculate_statistics(dataset_class, root):
    """
    Calculate statistics for small datasets, that can fit in memory.

    When adapting to larger datasets, careful with batching -
    average of standard deviations for each batch is not equal
    to the standard deviation of the dataset.

    See discussion in:
    https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks
    """
    with FileLock(f"{root}.lock"):
        dataset = dataset_class(root=root, train=True, download=True)

    loader = DataLoader(dataset, batch_size=sys.maxsize)
    data, _ = next(iter(loader))
    return (
        torch.mean(data, axis=[0, 2, 3]).totuple(),
        torch.std(data, axis=[0, 2, 3]).totuple()
    )
