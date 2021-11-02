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

import math
from collections import defaultdict

__all__ = [
    "PermutedMNISTTaskIndices",
]


class PermutedMNISTTaskIndices:
    """
    A mixin that overwrites `compute_task_indices` when using permutedMNIST to allow
    for much faster dataset initialization. Note that this mixin may not work with
    other datasets.
    """

    @classmethod
    def compute_task_indices(cls, config, dataset):
        # Assume dataloaders are already created
        class_indices = defaultdict(list)
        for idx in range(len(dataset)):
            target = _get_target(dataset, idx)
            class_indices[target].append(idx)

        # Defines how many classes should exist per task
        num_tasks = config.get("num_tasks", 1)
        num_classes = config.get("num_classes", None)
        assert num_classes is not None, "num_classes should be defined"
        num_classes_per_task = math.floor(num_classes / num_tasks)

        task_indices = defaultdict(list)
        for i in range(num_tasks):
            for j in range(num_classes_per_task):
                task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])
        return task_indices


def _get_target(dataset, idx):
    target = int(dataset.targets[idx % len(dataset.data)])
    task_id = dataset.get_task_id(idx)
    target += 10 * task_id
    return target
