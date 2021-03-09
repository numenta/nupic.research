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

import torch

from .base import TrackStatsHookBase


class TrackGradientsHook(TrackStatsHookBase):
    """
    Backward hook class for tracking gradients
    """

    def __init__(self, name, max_samples_to_track):
        super().__init__(name=name)

        self.num_samples = max_samples_to_track

        # `_gradients` keeps all gradients in memory, and grows linearly in
        # space with the number of samples
        self._gradients = torch.tensor([])

    def get_statistics(self):
        return [self._gradients]

    def __call__(self, module, grad_in, grad_out):
        """
        Backward hook on torch.nn.Module.

        :param module: module
        :param grad_in: gradient of model output wrt. layer output from forward pass
        :param grad_out: grad_in * (gradient of layer output wrt. layer input)
        """
        if not self._tracking:
            return
        self._gradients = self._gradients.to(grad_in[1].device)
        self._gradients = torch.cat((grad_in[1].reshape(1, -1), self._gradients), dim=0)
        # Keep only the last 'num_samples'
        self._gradients = self._gradients[: self.num_samples, ...]
