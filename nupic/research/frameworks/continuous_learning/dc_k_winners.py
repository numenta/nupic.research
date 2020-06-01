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

import nupic.torch.functions as F
from nupic.torch.modules.k_winners import KWinnersBase


def new_epoch(m):
    if isinstance(m, KWinnersBase):
        m.get_new_epoch()


def per_epoch(m, bpe):
    if isinstance(m, KWinnersBase):
        m.per_epoch = bpe


class DCKWinners(KWinnersBase):
    """Experimental; Applies K-Winner function to the input tensor
    with duty cycles updated between epochs (rather than between inputs)"

    See :class:`htmresearch.frameworks.pytorch.functions.k_winners`

    :param n:
      Number of units
    :type n: int

    :param percent_on:
      The activity of the top k = percent_on * n will be allowed to remain, the
      rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param new_epoch:
      If True and per_epoch=True, this will trigger update of the duty cycles
    :type new_epoch: bool

    :param per_epoch:
      If True, the module will update duty cycles whenever new_epoch=True
    :type per_epoch: bool
    """

    def __init__(
        self,
        n,
        percent_on,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        new_epoch=False,
        per_epoch=False,
    ):

        super(DCKWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            new_epoch=new_epoch,
        )
        self.n = n
        self.k = int(round(n * percent_on))
        self.k_inference = int(self.k * self.k_inference_factor)
        self.register_buffer("duty_cycle", torch.zeros(self.n))
        self.register_buffer("running_duty_cycle", self.duty_cycle)
        self.new_epoch = new_epoch
        self.per_epoch = per_epoch

    def forward(self, x):
        if self.training:
            x = F.KWinners.apply(x, self.running_duty_cycle,
                                 self.k, self.boost_strength)
            self.update_duty_cycle(x)
            print(self.per_epoch)
            if self.per_epoch:
                if self.new_epoch:
                    self.running_duty_cycle = self.duty_cycle
                    self.new_epoch = False
            else:
                self.running_duty_cycle = self.duty_cycle

        else:
            x = F.KWinners.apply(x, self.running_duty_cycle, self.k_inference,
                                 self.boost_strength)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        self.duty_cycle.add_(x.gt(0).sum(dim=0, dtype=torch.float))
        self.duty_cycle.div_(period)


class DCKWinners2d(KWinnersBase):
    """
    Experimental; Applies K-Winner function to the input tensor
    with duty cycles updated between epochs (rather than between inputs)"

    See :class:`htmresearch.frameworks.pytorch.functions.k_winners2d`

    :param channels:
      Number of channels (filters) in the convolutional layer.
    :type channels: int

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param k_inference_factor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * k_inference_factor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type k_inference_factor: float

    :param boost_strength:
      boost strength (0.0 implies no boosting).
    :type boost_strength: float

    :param boost_strength_factor:
      Boost strength factor to use [0..1]
    :type boost_strength_factor: float

    :param duty_cycle_period:
      The period used to calculate duty cycles
    :type duty_cycle_period: int

    :param local:
        Whether or not to choose the k-winners locally (across the channels
        at each location) or globally (across the whole input and across
        all channels).
    :type local: bool

    :param new_epoch:
      If True and per_epoch=True, this will trigger update of the duty cycles
    :type new_epoch: bool

    :param per_epoch:
      If True, the module will update duty cycles whenever new_epoch=True
    :type per_epoch: bool
    """

    def __init__(
        self,
        channels,
        percent_on=0.1,
        k_inference_factor=1.5,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        duty_cycle_period=1000,
        local=False,
        new_epoch=False,
        per_epoch=False,
    ):

        super(DCKWinners2d, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.channels = channels
        self.local = local
        if local:
            self.k = int(round(self.channels * self.percent_on))
            self.k_inference = int(round(self.channels * self.percent_on_inference))
            self.kwinner_function = F.KWinners2dLocal.apply
        else:
            self.kwinner_function = F.KWinners2dGlobal.apply

        self.register_buffer("duty_cycle", torch.zeros((1, channels, 1, 1)))
        self.register_buffer("running_duty_cycle", self.duty_cycle)
        self.new_epoch = new_epoch
        self.per_epoch = per_epoch

    def forward(self, x):

        if self.n == 0:
            self.n = np.prod(x.shape[1:])
            if not self.local:
                self.k = int(round(self.n * self.percent_on))
                self.k_inference = int(round(self.n * self.percent_on_inference))

        if self.training:
            x = self.kwinner_function(x, self.running_duty_cycle, self.k,
                                      self.boost_strength)
            self.update_duty_cycle(x)
            if self.per_epoch:
                if self.new_epoch:
                    self.running_duty_cycle = self.duty_cycle
                    self.new_epoch = False
            else:
                self.running_duty_cycle = self.duty_cycle

        else:
            x = self.kwinner_function(x, self.running_duty_cycle, self.k_inference,
                                      self.boost_strength)

        return x

    def update_duty_cycle(self, x):
        batch_size = x.shape[0]
        self.learning_iterations += batch_size

        scale_factor = float(x.shape[2] * x.shape[3])
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycle.mul_(period - batch_size)
        s = x.gt(0).sum(dim=(0, 2, 3), dtype=torch.float) / scale_factor
        self.duty_cycle.reshape(-1).add_(s)
        self.duty_cycle.div_(period)

    def entropy(self):
        entropy = super(DCKWinners2d, self).entropy()
        return entropy * self.n / self.channels

    def extra_repr(self):
        return "channels={}, local={}, {}".format(
            self.channels, self.local, super(DCKWinners2d, self).extra_repr()
        )
