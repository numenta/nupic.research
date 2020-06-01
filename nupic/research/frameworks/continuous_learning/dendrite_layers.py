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

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dend_kwinners import DKWinners
from nupic.torch.modules import SparseWeights  # , KWinnersBase, KWinners


class DendriteInput(nn.Module):
    def __init__(self,
                 in_dim,
                 n_dendrites,
                 threshold=2,
                 sparse_weights=True,
                 weight_sparsity=0.2,
                 ):
        super(DendriteInput, self).__init__()
        self.threshold = threshold
        linear = nn.Linear(in_dim, n_dendrites)

        if sparse_weights:
            self.linear = SparseWeights(linear, weight_sparsity)
        else:
            self.linear = linear

    def dendrite_activation(self, x):
        return torch.clamp(x, min=self.threshold)

    def forward(self, x):
        out = self.linear(x)
        return out  # self.act_fun(out)


class DendriteOutput(nn.Module):
    def __init__(self, out_dim, dpc):
        super(DendriteOutput, self).__init__()
        self.dpc = dpc
        self.register_buffer("mask", self.dend_mask(out_dim))
        self.weight = torch.nn.Parameter(torch.Tensor(out_dim, dpc * out_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_normal_(self.weight)
        self.bias.data.fill_(0.)

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def dend_mask(self, out_dim):
        mask = torch.zeros(out_dim, out_dim)
        inds = np.diag_indices(out_dim)
        mask[inds[0], inds[1]] = 1.
        out_mask = torch.repeat_interleave(mask, self.dpc, dim=0).T
        return out_mask


class DendriteLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dpc,
                 threshold=2,
                 sparse_weights=True,
                 weight_sparsity=0.2,
                 k_inference_factor=1.,
                 boost_strength=2.,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 ):
        super(DendriteLayer, self).__init__()

        self.dpc = dpc
        self.n_dendrites = out_dim * self.dpc
        self.out_dim = out_dim

        self.threshold = threshold
        self.input = DendriteInput(
            in_dim=in_dim,
            n_dendrites=self.n_dendrites,
            threshold=self.threshold,
            sparse_weights=sparse_weights,
            weight_sparsity=weight_sparsity,
        )
        self.output = DendriteOutput(out_dim, self.dpc)

        self.act_fun = DKWinners(
            n=self.n_dendrites,
            out_dim=self.out_dim,
            dpc=self.dpc,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

    def forward(self, x):
        out1 = self.act_fun(self.input(x))
        out2 = self.output(out1)
        return out2
