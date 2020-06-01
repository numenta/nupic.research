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

from nupic.torch.modules import SparseWeights  # , KWinnersBase, KWinners
from nupic.torch.modules.k_winners import KWinners2d


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

    def __init__(self,
                 in_dim,
                 n_dendrites,
                 threshold=2,
                 weight_sparsity=0.2,
                 ):
        super(DendriteInput, self).__init__()
        self.threshold = threshold
        linear = nn.Linear(in_dim, n_dendrites)

        if weight_sparsity < 1:
            self.linear = SparseWeights(linear, weight_sparsity)
        else:
            self.linear = linear

    def dendrite_activation(self, x):
        return torch.clamp(x, min=self.threshold)

    def forward(self, x):
        out = self.linear(x)
        return out  # self.act_fun(out)


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
        self.weight = torch.nn.Parameter(torch.Tensor(out_dim,
                                                      dendrites_per_unit * out_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(out_dim))
        # for stability - will integrate separate weight init. later
        nn.init.xavier_normal_(self.weight)
        self.bias.data.fill_(0.)

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)

    def dend_mask(self, out_dim):
        """This creates a mask such that each dendrite
        unit only projects to one downstream unit
        """
        mask = torch.zeros(out_dim, out_dim)
        inds = np.diag_indices(out_dim)
        mask[inds[0], inds[1]] = 1.
        out_mask = torch.repeat_interleave(mask, self.dendrites_per_unit, dim=0).T
        return out_mask


class DendriteLayer(nn.Module):
    """ This is the full module, combining DendriteInput
    and DendriteOutput. The module also specifies an
    activation function for the dendrite units
    (in this case a Kwinners2DLocal).

    The parameters k_inference_factor through
    duty_cycle_period are parameters for the KWinner2D
    activation. See 'nupic.torch.modules.k_winners'
    Note that "percent_on" will be overwritten in the
    KWinner2D module to specify k=1.

    :param in_dim: input dimension for DendriteInput

    :param out_dim: output dimension for DendriteOutput

    :param dendrites_per_neuron: dendrites per downstream unit


    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 dendrites_per_neuron,
                 weight_sparsity=0.2,
                 k_inference_factor=1.,
                 boost_strength=2.,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 ):
        super(DendriteLayer, self).__init__()

        self.dendrites_per_neuron = dendrites_per_neuron
        self.n_dendrites = out_dim * self.dendrites_per_neuron
        self.out_dim = out_dim

        self.input = DendriteInput(
            in_dim=in_dim,
            n_dendrites=self.n_dendrites,
            weight_sparsity=weight_sparsity,
        )
        self.output = DendriteOutput(out_dim, self.dendrites_per_neuron)

        self.act_fun = KWinners2d(
            channels=self.out_dim,
            percent_on=0.1,  # will be overwritten
            k_inference_factor=k_inference_factor,  # will be overwritten
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            local=True,
        )
        self.act_fun.k = 1
        self.act_fun.k_inference = 1

    def forward(self, x):
        batch_size = x.shape[0]
        out0 = self.input(x)
        with torch.no_grad():
            out0_ = out0.reshape(batch_size, self.dpc, self.out_dim, 1)

        out1 = self.act_fun(out0_)
        with torch.no_grad():
            out1_ = torch.squeeze(out1)
        out1_ = out1_.reshape(batch_size, self.out_dim * self.dpc)

        out2 = self.output(out1_)
        return out2
