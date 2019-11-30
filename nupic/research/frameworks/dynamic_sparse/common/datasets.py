# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
from collections.abc import Iterable

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nupic.research.frameworks.dynamic_sparse.common.dataloaders import (
    PreprocessedSpeechDataLoader,
    VaryingDataLoader,
)
from nupic.research.frameworks.pytorch.dataset_utils import CachedDatasetFolder
from nupic.research.frameworks.pytorch.image_transforms import RandomNoise
from nupic.research.frameworks.pytorch.tiny_imagenet_dataset import TinyImageNet

custom_datasets = {"TinyImageNet": TinyImageNet}

datasets_numclasses = {"TinyImageNet": 200, "CIFAR10": 10, "CIFAR100": 100, "MNIST": 10}

datasets_stats = {
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "TinyImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": (
        (0.50707516, 0.48654887, 0.44091784),
        (0.26733429, 0.25643846, 0.27615047),
    ),
    "MNIST": ((0.13062755,), (0.30810780,)),
}


def load_dataset(dataset_name):
    if dataset_name == "PreprocessedGSC":
        return GSCDataset
    elif dataset_name == "ImageNet":
        return ImageNetDataset
    else:
        return BaseDataset


class BaseDataset:
    """Loads a dataset.
    Returns object with a pytorch train and test loader
    """

    def __init__(self, config=None):

        defaults = dict(
            dataset_name=None,
            data_dir="~/nta/datasets",
            batch_size_train=128,
            batch_size_test=128,
            stats_mean=None,
            stats_std=None,
            augment_images=False,
            test_noise=False,
            noise_level=0.1,
        )
        defaults.update(config)
        self.__dict__.update(defaults)
        self.data_dir = os.path.expanduser(self.data_dir)
        self.load_dataset()

    def calc_statistics(self):

        tempset = self.dataset(
            root=self.data_dir, train=True, transform=transforms.ToTensor()
        )
        if isinstance(tempset.data, np.ndarray):
            stats_mean = (tempset.data.mean() / 255,)
            stats_std = (tempset.data.std() / 255,)
        else:
            stats_mean = (tempset.data.float().mean().item() / 255,)
            stats_std = (tempset.data.float().std().item() / 255,)
        del tempset  # explicit gc to avoid memory leakage

        return stats_mean, stats_std

    def load_dataset(self):

        # allow for custom datasets
        if self.dataset_name in custom_datasets:
            self.dataset = custom_datasets[self.dataset_name]
        elif hasattr(datasets, self.dataset_name):
            self.dataset = getattr(datasets, self.dataset_name)
        else:
            raise Exception("Dataset {} not available".format(self.dataset_name))

        # special dataloader case
        if isinstance(self.batch_size_train, Iterable) or isinstance(
            self.batch_size_test, Iterable
        ):
            dataloader_type = VaryingDataLoader
        else:
            dataloader_type = DataLoader

        # expand ~
        self.data_dir = os.path.expanduser(self.data_dir)

        # calculate statistics only if not already stored
        if self.dataset_name not in datasets_stats:
            self.stats_mean, self.stats_std = self.calc_statistics()
        else:
            self.stats_mean, self.stats_std = datasets_stats[self.dataset_name]

        # set up transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.stats_mean, self.stats_std),
            ]
        )
        # set up augment transforms for training
        if not self.augment_images:
            aug_transform = transform
        else:
            aug_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.stats_mean, self.stats_std),
                ]
            )

        # load train set
        train_set = self.dataset(
            root=self.data_dir, train=True, transform=aug_transform
        )
        self.train_loader = dataloader_type(
            dataset=train_set, batch_size=self.batch_size_train, shuffle=True
        )

        # load test set
        test_set = self.dataset(root=self.data_dir, train=False, transform=transform)
        self.test_loader = dataloader_type(
            dataset=test_set, batch_size=self.batch_size_test, shuffle=False
        )

        # noise dataset
        if self.test_noise:
            self.set_noise_loader(self.noise_level)

    def set_noise_loader(self, noise):
        """Defines noise loader"""
        noise_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.stats_mean, self.stats_std),
                RandomNoise(noise, high_value=0.5 + 2 * 0.20, low_value=0.5 - 2 * 0.2),
            ]
        )
        noise_set = self.dataset(
            root=self.data_dir, train=False, transform=noise_transform
        )
        self.noise_loader = DataLoader(
            dataset=noise_set, batch_size=self.batch_size_test, shuffle=False
        )


class ImageNetDataset(BaseDataset):
    def load_dataset(self):
        """
        Overrides base dataset loading
        Fixes path to data
        Fixes all tranformations to be identical
        Preprocessing from: https://github.com/pytorch/vision/issues/39
        """

        train_path = os.path.expanduser("~/nta/data/imagenet/train")
        val_path = os.path.expanduser("~/nta/data/imagenet/val")

        stats_mean, stats_std = datasets_stats["ImageNet"]
        train_transform = transforms.Compose(
            [
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stats_mean, stats_std),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(stats_mean, stats_std),
            ]
        )

        # load datasets
        train_dataset = CachedDatasetFolder(
            train_path, transform=train_transform, num_classes=self.num_classes
        )
        test_dataset = CachedDatasetFolder(
            val_path, transform=val_transform, num_classes=self.num_classes
        )

        # load dataloaders
        # added pin_memory=True for faster data recovery
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size_train,
            pin_memory=True,
            num_workers=56,
        )
        self.test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.batch_size_test,
            pin_memory=True,
            num_workers=56,
        )


class GSCDataset(BaseDataset):
    def load_dataset(self):
        """ Overrides base dataset loading"""

        self.train_loader = PreprocessedSpeechDataLoader(
            self.data_dir,
            subset="train",
            batch_size=self.batch_size_train,
            shuffle=True,
        )

        self.test_loader = PreprocessedSpeechDataLoader(
            self.data_dir, subset="valid", batch_size=self.batch_size_test
        )

        if self.test_noise:
            self.noise_loader = PreprocessedSpeechDataLoader(
                self.data_dir,
                subset="test_noise",
                noise_level=self.noise_level,
                batch_size=self.batch_size_test,
            )
        else:
            self.noise_loader = None


class CustomDataset:
    def __init__(self, config=None):
        pass

    def set_loaders(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
