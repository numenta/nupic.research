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

class OmniglotDataset(object):

    def __init__(self, data_dir, sample_transform=None, target_transform=None):
        """
        Regular Torchvision dataset. 

        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        """

        # defaults to base transform
        if sample_transform is None:
            transform = self.base_transform()

        # TODO: rename data to dataset_dir
        dataset_class = datasets.Omniglot
        self.train_dataset = dataset_class(
            root=os.path.expanduser(data_dir),
            background=True,
            transform=transform,
            target_transform=target_transform,
            download=False,
        )

        self.val_dataset = dataset_class(
            root=os.path.expanduser(data_dir),
            background=True,
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
    def base_transform(cls):
        """Convert to tensor. Omniglot is already normalized"""
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
