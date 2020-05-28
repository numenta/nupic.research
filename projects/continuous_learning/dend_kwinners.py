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

import torch
import torch.nn as nn

from nupic.torch.duty_cycle_metrics import binary_entropy, max_entropy


def update_boost_strength(m):
    """Function used to update KWinner modules boost strength. This is typically done
    during training at the beginning of each epoch.

    Call using :meth:`torch.nn.Module.apply` after each epoch if required
    For example: ``m.apply(update_boost_strength)``

    :param m: KWinner module
    """
    if isinstance(m, DKWinnersBase):
        m.update_boost_strength()


class DKWinnersBase(nn.Module, metaclass=abc.ABCMeta):

    def __init__(
        self,
        out_dim=10,
        dpc=3,
        k_inference_factor=1.0,
        boost_strength=1.0,
        boost_strength_factor=1.0,
        duty_cycle_period=1000,
        new_epoch=False,
        per_epoch=False,
    ):
        super(DKWinnersBase, self).__init__()
        assert boost_strength >= 0.0
        assert 0.0 <= boost_strength_factor <= 1.0

        self.k_inference_factor = k_inference_factor
        self.learning_iterations = 0
        self.n = 0
        self.k = 0
        self.k_inference = 0
        self.dpc = dpc,

        # Boosting related parameters
        self.register_buffer("boost_strength", torch.tensor(boost_strength,
                                                            dtype=torch.float))
        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period
        self.new_epoch = new_epoch
        self.per_epoch = per_epoch

    def extra_repr(self):
        return (
            "n={0}, dpc={1}, boost_strength={2}, boost_strength_factor={3}, "
            "k_inference_factor={4}, duty_cycle_period={5}".format(
                self.n, self.dpc, self.boost_strength,
                self.boost_strength_factor, self.k_inference_factor,
                self.duty_cycle_period
            )
        )

    @abc.abstractmethod
    def update_duty_cycle(self, x):
        r"""Updates our duty cycle estimates with the new value. Duty cycles are
        updated according to the following formula:

        .. math::
            dutyCycle = \frac{dutyCycle \times \left( period - batchSize \right)
                                + newValue}{period}

        :param x:
          Current activity of each unit
        """
        raise NotImplementedError

    def update_boost_strength(self):
        """Update boost strength by multiplying by the boost strength factor.
        This is typically done during training at the beginning of each epoch.
        """
        self.boost_strength *= self.boost_strength_factor

    def entropy(self):
        """Returns the current total entropy of this layer."""
        _, entropy = binary_entropy(self.duty_cycle)
        return entropy

    def max_entropy(self):
        """Returns the maximum total entropy we can expect from this layer."""
        return max_entropy(self.n, int(self.n * self.percent_on))


class DKWinners(DKWinnersBase):
    def __init__(
        self,
        n,
        out_dim,
        dpc,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
    ):
        super(DKWinners, self).__init__(
            out_dim=out_dim,
            dpc=dpc,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )
        self.n = n
        self.k = out_dim  # int(round(n * percent_on))
        self.k_inference = int(self.k * self.k_inference_factor)
        self.register_buffer("duty_cycle", torch.zeros(self.n))

        if type(self.dpc) == tuple:
            self.dpc = self.dpc[0]

    def forward(self, x):

        if self.training:
            x = GenericKWinners.apply(x, self.duty_cycle, self.k,
                                      self.dpc, self.boost_strength)
            self.update_duty_cycle(x)
        else:
            x = GenericKWinners.apply(x, self.duty_cycle, self.k_inference,
                                      self.dpc, self.boost_strength)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        self.duty_cycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.duty_cycle.div_(period)


class GenericKWinners(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, duty_cycles, out_dim, dpc, boost_strength):

        k = out_dim

        if boost_strength > 0.0:
            target_density = float(k) / x.size(1)
            boost_factors = torch.exp((target_density - duty_cycles) * boost_strength)
            boosted = x.detach() * boost_factors
        else:
            boosted = x.detach()

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
