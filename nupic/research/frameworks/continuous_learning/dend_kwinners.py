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

import abc

import numpy as np
import torch
import torch.nn as nn


class DKWinnersBase(nn.Module, metaclass=abc.ABCMeta):
    """ Just enough of the KWinnersBase to be able to put it into
    an experiment - no concept of duty cycles here, the idea is to do
    a maxout (i.e. k-winner with k=1) on dendrites per neuron.
    It uses some awkward matrix reshaping to do this, adapted from
    "KWinners2DLocal".

    Currently this is only implemented as a linear layer with a mask - see
    'nupic.research.frameworks.continuous_learning.dendrite_layers'

    :param dendrites_per_neuron: the number of dendrite segments per unit
    """
    def __init__(
        self,
        dendrites_per_neuron=3,
    ):
        super(DKWinnersBase, self).__init__()

        self.n = 0
        self.k = 0
        self.k_inference = 0
        self.dendrites_per_neuron = dendrites_per_neuron,

    def extra_repr(self):
        return (
            "n={0}, dendrites_per_neuron={1},".format(
                self.n, self.dendrites_per_neuron,
            )
        )


class DKWinners(DKWinnersBase):
    def __init__(
        self,
        n,
        out_dim,
        dendrites_per_neuron,
    ):
        super(DKWinners, self).__init__(
            out_dim=out_dim,
            dendrites_per_neuron=dendrites_per_neuron,
        )
        self.n = n
        self.k = out_dim

        if type(self.dendrites_per_neuron) == tuple:
            # Unexplained bug - need to fix
            self.dendrites_per_neuron = self.dendrites_per_neuron[0]

    def forward(self, x):
        x = DendriteKWinners.apply(x, self.k, self.dendrites_per_neuron)
        return x


class DendriteKWinners2d(DKWinnersBase):

    def __init__(
        self,
        channels,
        k=1,
    ):
        super(DendriteKWinners2d, self).__init__()

        self.channels = channels

        self.k = k
        # local only
        self.kwinner_function = DendriteKWinners2dLocal.apply

    def forward(self, x):
        if self.n == 0:
            self.n = np.prod(x.shape[1:])
            if not self.local:
                self.k = int(round(self.n * self.percent_on))

        x = self.kwinner_function(x, self.k)

        return x


class DendriteKWinners2dLocal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):

        batch_size, channels, h, w = x.shape
        boosted = x.detach()

        # Select top K channels from the boosted values
        topk, indices = boosted.topk(k=k, dim=1)
        res = torch.zeros_like(x)
        res.scatter_(1, indices, x.gather(1, indices))

        ctx.save_for_backward(indices)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we set the gradient to 1 for the winning units,
        and 0 for the others.
        """
        indices, = ctx.saved_tensors
        grad_x = torch.zeros_like(grad_output, requires_grad=False)
        grad_x.scatter_(1, indices, grad_output.gather(1, indices))
        return grad_x, None, None, None, None


class DendriteKWinners(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_dim, dendrites_per_neuron):

        k = out_dim
        boosted = x.detach()
        dpc = dendrites_per_neuron  # for flake8 reasons
        # Create a mask that takes the most active out of the N dendrites for each unit
        mask = torch.zeros(boosted.shape[0], out_dim, dpc)
        for k in range(out_dim):
            ind = torch.argmax(boosted[:, k * (dpc - 1):k * (dpc - 1) + dpc], dim=1)
            mask[:, k, ind].fill_(1.)

        mask = mask.reshape(boosted.shape[0], out_dim * dpc).cuda()  # reshape
        res = mask * x
        ctx.save_for_backward(mask)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass, we set the gradient to 1 for the winning
        units, and 0 for the others.
        """
        mask, = ctx.saved_tensors
        grad_x = grad_output * mask
        grad_x.requires_grad_(True)
        return grad_x, None, None, None, None
