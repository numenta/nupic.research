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

import math

import torch
from torch.utils.data import DistributedSampler, Sampler
import numpy as np

from collections.abc import Iterable

__all__ = [
    "TaskDistributedSampler",
    "TaskRandomSampler",
]

class TaskDistributedSampler(DistributedSampler):

    def __init__(
        self, dataset, task_indices, num_replicas=None, rank=None, shuffle=True
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.task_indices = task_indices
        self.set_active_tasks(0)

    def set_active_tasks(self, task_indices):
        """Accepts index for task or list of indices"""
        print(f"Training, setting active task to {task_indices}")
        self.active_tasks = task_indices
        if not isinstance(self.active_tasks, Iterable):
            self.active_tasks = [task_indices]
        self.indices = np.concatenate([self.task_indices[t] for t in self.active_tasks])
        self.task_length = len(self.indices)
        self.num_samples = math.ceil(self.task_length * 1.0 / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = self.indices
        if self.shuffle:
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g)]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        """Returns the length of the active task"""
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



class TaskRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, task_indices):
        self.task_indices = task_indices
        self.set_active_tasks(0)

    def set_active_tasks(self, task_indices):
        print(f"Setting active task to {task_indices}")
        self.active_tasks = task_indices
        if not isinstance(self.active_tasks, Iterable):
            self.active_tasks = [task_indices]
        self.indices = np.concatenate([self.task_indices[t] for t in self.active_tasks])
        print(self.indices)

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
