# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import torch
from torchvision import transforms
from torchvision.datasets import MNIST


class PermutedMNIST(MNIST):
    """
    The permutedMNIST dataset contains MNIST images where the same random permutation
    of pixel values is applied to each image. More specifically, the dataset can be
    broken down into 'tasks', where each such task is the set of all MNIST images, but
    with a unique pixel-wise permutation applied to all images. `num_tasks` gives the
    number of 10-way classification tasks (each corresponding to a unique pixel-wise
    permutation) that a continual learner will try to learn in sequence.
    """

    def __init__(self, num_tasks, seed, train, root=".", target_transform=None,
                 download=False):

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13062755,), (0.30810780,)),
        ])
        super().__init__(root=root, train=train, transform=data_transform,
                         target_transform=target_transform, download=download)

        self.num_tasks = num_tasks

        # Use a generator object to manually set the seed and generate the same
        # num_tasks random permutations for both training and validation datasets; the
        # first one is the identity permutation (i.e., regular MNIST), represented
        # below as `None`
        g = torch.manual_seed(seed)

        self.permutations = [
            torch.randperm(784, generator=g) for task_id in range(1, num_tasks)
        ]
        self.permutations.insert(0, None)

    def __getitem__(self, index):
        """
        Returns an (image, target) pair.

        In particular, this method retrieves an MNIST image, and based on the value of
        `index`, determines which pixel-wise permutation to apply. Target values are
        also scaled to be unique to each permutation.
        """
        img, target = super().__getitem__(index % len(self.data))

        # Determine which task `index` corresponds to
        task_id = self.get_task_id(index)

        # Apply permutation to `img`
        img = permute(img, self.permutations[task_id])

        # Since target values are not shared between tasks, `target` should be in the
        # range [0 + 10 * task_id, 9 + 10 * task_id]
        target += 10 * task_id
        return img, target

    def __len__(self):
        return self.num_tasks * len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", "processed")

    def get_task_id(self, index):
        return index // len(self.data)


class ContextDependentPermutedMNIST(PermutedMNIST):
    """
    A variant of permutedMNIST where each permutation (i.e., 'task') is associated with
    a context: a binary sparse vector. The `__getitem__` method returns the context
    vector along with the data sample and target.
    """

    def __init__(self, num_tasks, dim_context, seed, train, root=".",
                 target_transform=None, download=False):

        super().__init__(num_tasks, seed, train, root, target_transform, download)
        self.dim_context = dim_context

        # Initialize random binary sparse context vectors for each permutation
        self.init_contexts(seed)

    def __getitem__(self, index):
        """
        Returns an ((image, context), target) tuple.
        """
        img, target = super().__getitem__(index)
        task_id = self.get_task_id(index)
        return (img, self.contexts[task_id, :]), target

    def init_contexts(self, seed):
        percent_on = 0.05
        num_contexts = self.num_tasks

        num_ones = int(percent_on * self.dim_context)
        num_zeros = self.dim_context - num_ones

        self.contexts = torch.cat((
            torch.zeros((num_contexts, num_zeros)),
            torch.ones((num_contexts, num_ones))
        ), dim=1)

        # Shuffle each context vector i.i.d. to randomize it; use a fixed seed during
        # shuffling so that train & validation contexts are identical
        g = torch.manual_seed(seed)

        for i in range(num_contexts):
            self.contexts[i, :] = self.contexts[i, torch.randperm(self.dim_context,
                                                                  generator=g)]


def permute(x, permutation):
    """
    Applies the permutation specified by `permutation` to `x` and returns the resulting
    tensor.
    """

    # Assume x has shape (1, height, width)
    if permutation is None:
        return x

    _, height, width = x.size()
    x = x.view(-1, 1)
    x = x[permutation, :]
    x = x.view(1, height, width)
    return x
