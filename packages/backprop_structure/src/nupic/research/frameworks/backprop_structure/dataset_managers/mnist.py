# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from nupic.research.frameworks.pytorch.image_transforms import RandomNoise

MEAN = 0.13062755
STDEV = 0.30810780


class CachingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = [None] * len(dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            self.cache[index] = self.dataset[index]
        return self.cache[index]

    def __len__(self):
        return len(self.dataset)


class MNIST(object):
    def __init__(self):
        self.folder = os.path.expanduser("~/nta/datasets")
        self.train_dataset = None
        self.test_datasets = {}

    def get_train_dataset(self, iteration):
        if self.train_dataset is None:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((MEAN,), (STDEV,))])
            self.train_dataset = CachingDataset(datasets.MNIST(
                self.folder, train=True, download=True, transform=transform))

        return self.train_dataset

    def get_test_dataset(self, noise_level=0.0):
        if noise_level not in self.test_datasets:
            all_transforms = [transforms.ToTensor()]
            if noise_level > 0.0:
                all_transforms.append(RandomNoise(noise_level))
            all_transforms.append(transforms.Normalize((MEAN,), (STDEV,)))

            transform = transforms.Compose(all_transforms)
            self.test_datasets[noise_level] = CachingDataset(datasets.MNIST(
                self.folder, train=False, download=True, transform=transform))

        return self.test_datasets[noise_level]
