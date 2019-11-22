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

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

MEAN = 0.13062755
STDEV = 0.30810780


class RandomNoise(object):
    """
    An image transform that adds noise to random pixels in the image.
    """
    def __init__(self, noise_level=0.0, white_value=(MEAN + (2 * STDEV))):
        """
        :param noise_level:
          From 0 to 1. For each pixel, set its value to white_value with this
          probability. Suggested white_value is 'mean + 2*stdev'
        """
        self.noise_level = noise_level
        self.white_value = white_value

    def __call__(self, image):
        a = image.view(-1)
        num_noise_bits = int(a.shape[0] * self.noise_level)
        noise = np.random.permutation(a.shape[0])[0:num_noise_bits]
        a[noise] = self.white_value
        return image


class MNIST(object):
    def __init__(self):
        self.folder = os.path.expanduser("~/nta/datasets")
        self.train_dataset = None

    def get_train_dataset(self, iteration):
        # The training set can be cached since it is the same every iteration.
        if self.train_dataset is None:
            transform = transforms.Compose([transforms.ToTensor()])
            self.train_dataset = datasets.MNIST(
                self.folder, train=True, download=True, transform=transform)

        return self.train_dataset

    def get_test_dataset(self, noise_level=0.0):
        all_transforms = [transforms.ToTensor()]
        if noise_level > 0.0:
            all_transforms += [RandomNoise(noise_level),
                               transforms.Normalize((MEAN,), (STDEV,))]
        transform = transforms.Compose(all_transforms)
        return datasets.MNIST(
            self.folder, train=False, download=True, transform=transform)
