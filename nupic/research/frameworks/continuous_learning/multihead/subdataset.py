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

"""
Dataset functionality for MNIST and Google Speech Commands in a continual learning
framework.

This implementation is based on the original continual learning benchmarks repository:
https://github.com/GMvandeVen/continual-learning
"""

import os
import warnings

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


def get_datasets(dataset_name, scenario, download=True):
    """
    Return the train and test datasets (in parallel lists) for each task in the
    continuous learning setup
    :param dataset_name: name of the dataset to retrieve
    :type dataset_name: str
    :param scenario: continuous learning setup, one of {'task', 'domain', 'class'}
    :type scenario: str
    :param download: whether or not to download the dataset
    :type download: bool
    :return: tuple of (list of pytorch dataset, list of pytorch dataset)
    """
    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if dataset_name == "splitGSC":
        dataset_class = GSC
    elif dataset_name == "splitMNIST":
        dataset_class = MNIST

    # In the following class initializations, the `root` directory may need to be
    # modified based on where MNIST/GSC data is stored
    full_train_set = dataset_class(".",
                                   train=True, transform=dataset_transform,
                                   target_transform=None, download=download)
    full_test_set = dataset_class(".",
                                  train=False, transform=dataset_transform,
                                  target_transform=None, download=download)

    train_datasets, test_datasets = [], []

    for labels in [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]:

        if scenario in ("task", "domain"):
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x)
        else:
            target_transform = None

        train_datasets.append(SubDataset(full_train_set, labels,
                                         target_transform=target_transform))
        test_datasets.append(SubDataset(full_test_set, labels,
                                        target_transform=target_transform))

    return train_datasets, test_datasets


class GSC(Dataset):
    """ Google Speech Commands dataset. """

    def __init__(self, root=".", train=True, transform=None, target_transform=None,
                 download=False):
        # The `download` parameter is only for consistency with the torchvision API
        # and will issue a warning if set to True
        if download:
            warnings.warn(
                "Parameter `download` is deprecated is only for consistency with the\
                torchvision API and will not be used.",
                DeprecationWarning,
            )

        super(GSC, self).__init__()

        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.data = [np.load(os.path.join(
                root, "gsc_train{}.npz".format(n)
            ))["arr_0"] for n in range(30)]
            self.data = np.concatenate(self.data)

            self.targets = [np.load(os.path.join(
                root, "gsc_train{}.npz".format(n)
            ))["arr_1"] for n in range(30)]
            self.targets = np.concatenate(self.targets)

            # remove items with labels 0 or 1
            idx = (self.targets > 1)
            self.data = self.data[idx]
            self.targets = self.targets[idx]

            # shift all labels to be at 0
            self.targets = self.targets - 2

            # select subset of data
            inds = np.random.choice(np.arange(len(self.data)), size=60000,
                                    replace=False)
            self.data = self.data[inds]
            self.targets = self.targets[inds]

        else:
            self.data = np.concatenate((
                np.load(os.path.join(root, "gsc_valid.npz"))["arr_0"],
                np.load(os.path.join(root, "gsc_test_noise00.npz"))["arr_0"]
            ))
            self.test_labels = np.concatenate((
                np.load(os.path.join(root, "gsc_valid.npz"))["arr_1"],
                np.load(os.path.join(root, "gsc_test_noise00.npz"))["arr_1"]
            ))

            # remove items with labels 0 or 1
            idx = (self.test_labels > 1)
            self.data = self.data[idx]
            self.test_labels = self.test_labels[idx]

            # shift all labels to be at 0
            self.test_labels = self.test_labels - 2

    def __getitem__(self, index):
        sound, target = self.data[index], int(self.targets[index]) if\
            hasattr(self, "targets") else int(self.test_labels[index])

        if self.transform is not None:
            sound = self.transform(sound)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sound, target

    def __len__(self):
        return len(self.data)


class SubDataset(Dataset):
    """ A SubDataset class that divides a given dataset with labels into one with
    specified labels. Note that a SubDataset `target_transform` is separate from a
    Dataset's `target_transform`. """

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indices = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indices.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indices)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indices[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample
