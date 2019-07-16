# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
import torch
from torch import nn


class KWinners(nn.Module):
    """Test implementation of KWinners
    TODO:
    - T should be per sample, not per iteration

    Currently not being used
    """

    def __init__(self, k_perc=0.1, use_absolute=False, use_boosting=False, beta=0.01):
        super(KWinners, self).__init__()

        self.duty_cycle = None
        self.k_perc = k_perc
        self.beta = beta
        self.T = 600  # 1000
        self.current_time = 0
        self.use_absolute = use_absolute
        self.use_boosting = use_boosting

    def forward(self, x):

        # define k
        units_shape = x.shape[1:]  # remove batch size
        num_units = np.prod(units_shape)
        k = int(self.k_perc * num_units)

        # initialize duty cycle
        if self.duty_cycle is None:
            # for MLP alone
            self.duty_cycle = torch.zeros(units_shape)

        # keep track of number of past iteratctions
        if self.current_time < self.T:
            self.current_time += 1

        # calculating threshold and updating duty cycle
        tx = x.clone().detach()
        if self.use_absolute:
            tx = torch.abs(tx)
        # no need to calculate gradients
        with torch.set_grad_enabled(False):
            threshold = self._get_threshold(tx, k)
            # if duty cycle, apply boosting, at training only
            if self.duty_cycle is not None and self.training and self.use_boosting:
                boosting = self._calculate_boosting()
                tx *= boosting
            # get mask
            mask = tx > threshold

            # update duty cycle at training only
            if self.training:
                self._update_duty_cycle(mask)

        return x * mask.float()

    def _get_threshold(self, x, k):
        """Calculate dynamic theshold"""
        # k-winners over neurons only
        flatten_x = x.view(x.shape[0], -1)
        pos = int((flatten_x.shape[1] - k))
        threshold, _ = torch.kthvalue(flatten_x, pos, dim=-1)
        expanded_size = [x.shape[0]] + [1 for _ in range(len(x.shape) - 1)]
        return threshold.view(expanded_size)

    def _update_duty_cycle(self, mask):
        """Update duty cycle"""
        num_activations = torch.sum(mask, dim=0).float()
        time = min(self.T, self.current_time)
        self.duty_cycle *= (time - 1) / time
        self.duty_cycle += num_activations / time

    def _calculate_boosting(self):
        """Calculate boosting according to formula on spatial pooling paper"""
        mean_duty_cycle = torch.mean(self.duty_cycle)
        diff_duty_cycle = self.duty_cycle - mean_duty_cycle
        boosting = (self.beta * diff_duty_cycle).exp()
        # debug
        # b = boosting.view(-1)
        # print(torch.min(b).item(), torch.mean(b).item(), torch.max(b).item())
        return boosting
