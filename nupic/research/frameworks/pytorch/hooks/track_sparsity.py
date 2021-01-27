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

from .base import TrackStatsHookBase


class TrackSparsityHook(TrackStatsHookBase):
    """
    This forward hook tracks a cumulative mean for the sparsities observed in the input
    and output representations of the module. The measured sparsities can be accessed by
    calling `self.get_statistics()` and are reset once `self.start_tracking()` is
    called.

    :param name: (optional) name of the module (e.g. "classifier")
    """

    def __init__(self, name=None):
        super().__init__(name=name)

        self._input_sparsity = None
        self._output_sparsity = None
        self._total_samples = 0

    def get_statistics(self):
        return (self._input_sparsity, self._output_sparsity)

    def start_tracking(self):
        super().start_tracking()

        self._input_sparsity = 0
        self._output_sparsity = 0
        self._total_samples = 0

    def __call__(self, module, x, y):
        """
        Forward hook on torch.nn.Module.

        :param module: module
        :param x: tuple of inputs
        :param y: output of module
        """

        if not self._tracking:
            return

        if isinstance(x, tuple):
            x = x[0]
        if isinstance(y, tuple):
            y = y[0]

        # Track the running average sparsity of each x or y.
        batch_size = x.shape[0]
        self._input_sparsity *= self._total_samples
        self._input_sparsity += calculate_sparsities(x).sum().item()
        self._input_sparsity /= self._total_samples + batch_size

        self._output_sparsity *= self._total_samples
        self._output_sparsity += calculate_sparsities(y).sum().item()
        self._output_sparsity /= self._total_samples + batch_size

        self._total_samples += batch_size


def calculate_sparsities(rep):
    """
    Calculates the fraction of units off for each sample in the batch.

    :param rep: tensor of an intermediate representation
    :return: tensor of sparsities, one for each sample in the batch
    """
    rep = rep.flatten(start_dim=1)
    off_mask = rep == 0
    sparsities = off_mask.sum(axis=1).float()
    sparsities = sparsities / rep.shape[1]
    return sparsities
