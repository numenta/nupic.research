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

    def __init__(self, name, max_samples_to_track, metric="cosine"):
        super().__init__(name=name)

        self.num_samples = max_samples_to_track

        # `_gradients` keeps all hidden activations in memory, and grows linearly in
        # space with the number of samples
        # self._activations = None
        self._gradients = torch.tensor([])
        self.metric = metric

    def get_statistics(self):
        if self.metric == "cosine":
            return [torch.cosine_similarity(x, y) for x in self._gradients for y in
                    self._gradients]
        elif self.metric == "dot":
            return [x.dot(y) for x in self._gradients for y in self._gradients]
        return None

    def __call__(self, module, grad_in, grad_out):
        """
        Forward hook on torch.nn.Module.

        :param module: module
        :param grad_in: gradient of model output wrt. layer output from forward pass
        :param grad_out: grad_in * (gradient of layer output wrt. layer input)
        """
        if not self._tracking:
            return

        self._gradients = self._gradients.to(grad_out.device)
        self._gradients = torch.cat((grad_out, self._gradients), dim=0)

        # Keep only the last 'num_samples'
        self._gradients = self._gradients[:self.num_samples, ...]
