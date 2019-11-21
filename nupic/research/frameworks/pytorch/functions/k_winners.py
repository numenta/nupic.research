#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#


import torch


class KWinners2dLocal(torch.autograd.Function):
    """
    A K-winner take all autograd function for CNN 2D inputs (batch, Channel, H, W)
    where the k-winners are chosen locally across all the channels
    """

    @staticmethod
    def forward(ctx, x, duty_cycles, k, boost_strength):
        """
          Take the boosted version of the input x, find the top k winners across
          the channels. The output will only contain the values of x
          corresponding to the top k boosted values across all the channels.
          The rest of the elements in the output should be 0.

        :param ctx:
          Place where we can store information we will need to compute the
          gradients for the backward pass.

        :param x:
          Current activity of each unit.

        :param duty_cycles:
          The averaged duty cycle of each unit.

        :param k:
          The activity of the top k units across the channels will be allowed to
          remain, the rest are set to zero.

        :param boost_strength:
          A boost strength of 0.0 has no effect on x.

        :return:
             A tensor representing the activity of x after k-winner take all.
        """
        batch_size, channels, h, w = x.shape
        if boost_strength > 0.0:
            # Apply boost strength to input computing density per channel
            target_density = float(k) / channels
            boost_factors = torch.exp((target_density - duty_cycles) * boost_strength)
            boosted = x.detach() * boost_factors
        else:
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


def k_winners2d_local(x, duty_cycles, k, boost_strength):
    """
      Take the boosted version of the input x, find the top k winners across
      the channels. The output will only contain the values of x
      corresponding to the top k boosted values across all the channels.
      The rest of the elements in the output should be 0.

    :param x:
      Current activity of each unit.

    :param duty_cycles:
      The averaged duty cycle of each unit.

    :param k:
      The activity of the top k units across the channels will be allowed to
      remain, the rest are set to zero.

    :param boost_strength:
      A boost strength of 0.0 has no effect on x.

    :return:
         A tensor representing the activity of x after k-winner take all.
    """
    return KWinners2dLocal.apply(x, duty_cycles, k, boost_strength)
