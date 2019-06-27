# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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


class KWinnersMask(torch.autograd.Function):
    """A simple K-winner take all autograd function for creating layers with
    sparse output.

    .. note::
        Code adapted from this excellent tutorial:
        https://github.com/jcjohnson/pytorch-examples
    """

    @staticmethod
    def forward(ctx, x, k):
        r"""
        :param ctx:
          Place where we can store information we will need to compute the gradients
          for the backward pass.

        :param x:
          Current activity of each unit.

        :param k:
          The activity of the top k units will be allowed to remain, the rest are
          set to zero.

        :return:
          A tensor representing the activity of x after k-winner take all.
        """
        res = torch.zeros_like(x)
        topk, indices = x.topk(k, sorted=False)
        mask = res.scatter(-1, indices, 1)
        ctx.save_for_backward(indices)
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass, we set the gradient to 1 for the winning
        units, and 0 for the others.
        """
        indices, = ctx.saved_tensors
        grad_x = torch.zeros_like(grad_output, requires_grad=False)
        grad_x.scatter_(-1, indices, 1)
        return grad_x, None, None, None
