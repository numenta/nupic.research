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

from torchvision import datasets, transforms

DATASETS_STATS = {
    "ImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "TinyImageNet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "CIFAR100": (
        (0.50707516, 0.48654887, 0.44091784),
        (0.26733429, 0.25643846, 0.27615047),
    ),
    "MNIST": ((0.13062755,), (0.30810780,)),
}


class TorchvisionDataset(object):

    def __init__(self, data_dir, dataset_name, sample_transform=None, target_transform=None):

        # TODO: calculate statistics for any torchvision dataset, if not available
        if dataset_name not in DATASETS_STATS.keys():
            raise ValueError(f"{dataset_name} not available.")

        # defaults to base transform
        if sample_transform is None:
            transform = self.base_transform(*DATASETS_STATS[dataset_name])

        # TODO: rename data to dataset_dir
        dataset_class = getattr(datasets, dataset_name)
        self.train_dataset = dataset_class(
            root=os.path.expanduser(data_dir),
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

        self.val_dataset = dataset_class(
            root=os.path.expanduser(data_dir),
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return None

    @classmethod
    def base_transform(cls, stats_mean, stats_std):
        """Convert to tensor and normalize"""
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(stats_mean, stats_std),
            ]
        )
