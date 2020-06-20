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

from nupic.research.frameworks.continuous_learning.dend_kwinners import (
    DendriteKWinners2dLocal,
)
from nupic.torch.modules import SparseWeights


class DendriteInput(nn.Module):
    """ Sparse linear layer from previous output to
    "dendrites" - this is the first part of a module
    that projects via dendrite segments to output units.

    :param in_dim: input dimension

    :param n_dendrites: total number of dendrites - note this will
    be an integer multiple of the number of downstream units

    :param threshold: (currently unused) - threshold for an
    in-development dendritic activation or gating function

    :param weight_sparsity: Weight sparsity of the sparse weights.
    If weight_sparsity=1, it will default to a standard linear layer.
    """

    def __init__(
        self, in_dim, n_dendrites, threshold=2, weight_sparsity=0.2,
    ):
        super(DendriteInput, self).__init__()
        self.threshold = threshold
        linear = nn.Linear(in_dim, n_dendrites)

        if weight_sparsity < 1:
            self.linear = SparseWeights(linear, weight_sparsity)
        else:
            self.linear = linear

    def forward(self, x):
        """ Note this only returns the linear output """
        out = self.linear(x)
        return out


class DendriteOutput(nn.Module):
    """ Masked linear layer from dendrites to output
    units. This is the second part of the full module.

    :param out_dim: output dimension (number of downstream units)

    :param dendrites_per_unit: integer number of dendrite
    segments per unit
    """

    def __init__(self, out_dim, dendrites_per_unit):
        super(DendriteOutput, self).__init__()
        self.dendrites_per_unit = dendrites_per_unit
        self.register_buffer("mask", self.dend_mask(out_dim))
        self.weight = torch.nn.Parameter(
            torch.Tensor(out_dim, dendrites_per_unit * out_dim)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(out_dim), requires_grad=True)
        # for stability - will integrate separate weight init. later
        nn.init.kaiming_uniform_(self.weight)
        self.bias.data.fill_(0.0)

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def dend_mask(self, out_dim):
        """This creates a mask such that each dendrite
        unit only projects to one downstream unit
        """
        mask = torch.zeros(out_dim, out_dim)
        inds = np.diag_indices(out_dim)
        mask[inds[0], inds[1]] = 1.0
        out_mask = torch.repeat_interleave(mask, self.dendrites_per_unit, dim=0).T
        return out_mask


class DendriteLayer(nn.Module):
    """ This is the full module, combining DendriteInput
    and DendriteOutput. The module also specifies an
    activation function for the dendrite units
    (in this case a Kwinners2DLocal).

    :param in_dim: input dimension for DendriteInput

    :param out_dim: output dimension for DendriteOutput

    :param dendrites_per_neuron: dendrites per downstream unit

    :param weight_sparsity: DOC

    :param act_fun_type
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        dendrites_per_neuron,
        weight_sparsity=0.2,
        act_fun_type="kwinner",
    ):
        super(DendriteLayer, self).__init__()

        self.dendrites_per_neuron = dendrites_per_neuron
        self.n_dendrites = out_dim * self.dendrites_per_neuron
        self.out_dim = out_dim

        self.act_fun_type = act_fun_type

        self.input = DendriteInput(
            in_dim=in_dim,
            n_dendrites=self.n_dendrites,
            weight_sparsity=weight_sparsity,
        )
        self.output = DendriteOutput(out_dim, self.dendrites_per_neuron)

    def forward(self, x, cat_projection=1.0):
        """ cat_proj here is an optional argument
        for a categorical "feedback" projection to
        the dendrite segments
        """
        if self.act_fun_type == "kwinner":
            return self.forward_kwinner(x, cat_projection)
        elif self.act_fun_type == "sigmoid":
            return self.forward_sigmoid(x, cat_projection)
        else:
            raise AssertionError("act_fun_type must be ''kwinner'' or ''sigmoid'' ")

    def forward_kwinner(self, x, cat_projection=1.0):  # cat_projection = 1.0 is cleaner
        """ cat_projection is scalar categorical input
        """
        batch_size = x.shape[0]
        out0 = self.input(x)

        out0 = out0 * cat_projection  # will be identity without categorical projection
        # if statements introduce bug potential and are slower on GPU
        out0_ = out0.reshape(batch_size, self.out_dim, self.dendrites_per_neuron)

        out1_ = DendriteKWinners2dLocal.apply(out0_, 1)

        out1_1 = out1_.reshape(batch_size, self.out_dim * self.dendrites_per_neuron)

        out2 = self.output(out1_1)
        return out2

    def forward_sigmoid(self, x, cat_projection=1.0):
        out0 = self.input(x)

        out1_pre = out0 * cat_projection

        out1 = torch.sigmoid(out1_pre)

        out2 = self.output(out1)
        return out2
