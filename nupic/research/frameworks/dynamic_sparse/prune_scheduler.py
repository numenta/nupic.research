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

import abc

import numpy as np

__all__ = [
    "PruneSchedulerBase",
    "CosineDecayPruneScheduler",
]


class PruneSchedulerBase(object, metaclass=abc.ABCMeta):
    """
    This class calculates a pruning schedule through the methods
        * `get_prune_fraction`
        * `get_num_add`
    """
    def __init__(self):
        # Keep track of the number of cycles of pruning.
        self._step_count = 0

    @abc.abstractclassmethod
    def get_prune_fraction(self):
        """
        Retrieve percentage of weights to prune.

        :return: pruning rate (between 0 and 1)
        """
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_num_add(self, num_removed):
        """
        Retrieve number of weights to add following pruning.

        :param num_removed: number of param removed in pruning step

        :return type: int
        """
        raise NotImplementedError

    def step(self):
        self._step_count += 1


class CosineDecayPruneScheduler(PruneSchedulerBase):
    """
    This class enables a tapering of an initial pruning-rate while
    `get_num_add` always returns how much has been removed.

    :param prune_fraction: starting pruning rate between 0 and 1
    :param total_steps: total number of steps of training; this can be
                        training iterations or epochs depending on how often the user
                        calls self.step()
    :param warmup_steps: how many steps to wait before pruning starts
    """
    def __init__(self, prune_fraction, total_steps, warmup_steps=0):
        super().__init__()
        self.prune_fraction = prune_fraction
        self.total_steps = total_steps - warmup_steps
        self.warmup_steps = warmup_steps

    def get_prune_fraction(self):
        step_count = self._step_count - self.warmup_steps
        if step_count < 0:
            return 0

        return 0.5 * self.prune_fraction * (
            1 + np.cos((step_count * np.pi) / (self.total_steps - 1))
        )

    def get_num_add(self, num_removed):
        return num_removed
