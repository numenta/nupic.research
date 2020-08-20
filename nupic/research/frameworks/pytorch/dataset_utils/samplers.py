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

    def set_active_tasks(self, tasks):
        """Accepts index for task or list of indices"""
        # print(f"Setting active task to {tasks}")
        self.active_tasks = tasks
        if not isinstance(self.active_tasks, Iterable):
            self.active_tasks = [tasks]
        self.indices = np.concatenate([self.task_indices[t] for t in self.active_tasks])
        self.num_samples = math.ceil(len(self.indices) * 1.0 / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        # print(self.indices[:100])

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = self.indices
        if self.shuffle:
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g)]
        # print("inside one iteration")
        # print(indices[:100])

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

    def set_active_tasks(self, tasks):
        # print(f"Setting active task to {tasks}")
        self.active_tasks = tasks
        if not isinstance(self.active_tasks, Iterable):
            self.active_tasks = [tasks]
        self.indices = np.concatenate([self.task_indices[t] for t in self.active_tasks])
        # print(self.indices[:100])

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
