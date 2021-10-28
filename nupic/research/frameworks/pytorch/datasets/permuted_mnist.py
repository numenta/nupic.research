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
                 download=False, normalize=True):

        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize((0.13062755,), (0.30810780,)))
        data_transform = transforms.Compose(t)
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
    vector along with the data sample and target. Multiple types of context vectors are
    possible, and the context can be returned with data in a tuple, or concatenated to
    the data in which case the data are flattened first. This class assumes there is one
    context per task. The class below assumes each image is its own context, hence the
    decision to put it in a separate class.

    Unique to ContextDependentPermutedMNIST
        :param context_type: string indicating what kind of context vector to select.
                             Must be one of ["sparse_binary", "one_hot", "prototype"].
        :param combine_context_as: string indicating if context should be combined with
                                   data by packing both in a tuple, or concatenating
                                   context to the data. Must be one of
                                   ["tuple", "concatenate"]. Note that if concatenate is
                                   specified, data will be flattened!
    """

    def __init__(self, num_tasks, seed, train, context_type="sparse_binary",
                 combine_context_as="tuple", dim_context=None, root=".",
                 target_transform=None, download=False):

        super().__init__(num_tasks, seed, train, root, target_transform, download)
        self.dim_context = dim_context
        self.context_type = context_type

        # options for type of context and way of combining with input x
        context_type_choices = ["sparse_binary", "one_hot", "prototype"]
        combine_context_as_choices = ["tuple", "concatenate"]

        # Parse type of context
        if context_type == "sparse_binary":
            # Initialize random binary sparse context vectors for each permutation
            self.init_sparse_binary_contexts(seed)
        elif context_type == "one_hot":
            self.init_one_hot_contexts()
        elif context_type == "prototype":
            self.init_prototype_contexts()
        else:
            error_msg = f"context_type must be one of {context_type_choices}"
            raise ValueError(error_msg)

        # Parse how to combine image and context
        if combine_context_as == "tuple":
            self.combine_context = tuple_context
        elif combine_context_as == "concatenate":
            self.combine_context = concat_context
        else:
            error_msg = f"combine_context_as must be one of {combine_context_as_choices}"  # noqa E501
            raise ValueError(error_msg)

    def __getitem__(self, index):
        """
        Returns image, context, and target.
        """
        img, target = super().__getitem__(index)
        task_id = self.get_task_id(index)
        context = self.contexts[task_id, :]
        return self.combine_context(img, context), target

    def init_sparse_binary_contexts(self, seed):

        if self.dim_context is None:
            error_msg = "Please specify dim_context when using sparse_binary_context"
            raise ValueError(error_msg)

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

    def init_prototype_contexts(self):
        """
        Code to compute the mean image from each permutation. Note that you only need
        to compute the mean image for the base dataset. After that you can just apply
        each permutation to the mean vector.
        """
        self.dim_context = 784
        self.contexts = torch.zeros((self.num_tasks, 28, 28))
        for index in range(len(self.data)):
            img, _ = super().__getitem__(index)
            self.contexts[0] += img.squeeze(0)

        # This first row has the pixelwise sum for MNIST, divide to get mean
        self.contexts[0] /= len(self.data)

        # Now just apply permutations to the mean vector, one for each remaining task
        for task in range(1, self.num_tasks):
            self.contexts[task] = permute(self.contexts[0].unsqueeze(0),
                                          self.permutations[task])

        # 28 x 28 -> 784
        self.contexts = self.contexts.flatten(start_dim=1)

    def init_one_hot_contexts(self):
        self.dim_context = self.num_tasks
        self.contexts = torch.eye(self.num_tasks, self.num_tasks)


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


def tuple_context(x, context):
    return (x, context)


def concat_context(x, context):
    img = x.flatten()
    return torch.cat((img, context))
