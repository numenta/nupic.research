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

from torchvision import transforms
from torchvision.transforms import RandomResizedCrop

import nupic.research.frameworks.pytorch.imagenet.auto_augment as aa
from nupic.research.frameworks.pytorch.dataset_utils import HDF5Dataset
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import (
    IMAGENET_NUM_CLASSES as IMAGENET_CLASS_SUBSETS,
)


class ImageNet100(object):
    def __init__(self, use_auto_augment=False):
        self.use_auto_augment = use_auto_augment
        self.train_dataset = None
        self.test_dataset = None

    def get_train_dataset(self, iteration):
        if self.train_dataset is None:
            if self.use_auto_augment:
                transform = transforms.Compose(
                    transforms=[
                        RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        aa.ImageNetPolicy(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                            inplace=True
                        ),
                    ],
                )
            else:
                transform = transforms.Compose(
                    transforms=[
                        RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                            inplace=True
                        ),
                    ],
                )

            self.train_dataset = HDF5Dataset(
                hdf5_file=os.path.expanduser("~/nta/data/imagenet/imagenet.hdf5"),
                root="train",
                classes=IMAGENET_CLASS_SUBSETS[100],
                transform=transform)

        return self.train_dataset

    def get_test_dataset(self, noise_level=0.0):
        assert noise_level == 0.0

        if self.test_dataset is None:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                        inplace=True
                    ),
                ]
            )

            self.test_dataset = HDF5Dataset(
                hdf5_file=os.path.expanduser("~/nta/data/imagenet/imagenet.hdf5"),
                root="val",
                classes=IMAGENET_CLASS_SUBSETS[100],
                transform=transform)

        return self.test_dataset
