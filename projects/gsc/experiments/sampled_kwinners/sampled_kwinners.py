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
import abc

import numpy as np
import torch.nn as nn

import nupic.torch.functions as F
from nupic.torch.duty_cycle_metrics import binary_entropy, max_entropy


def update_temperature(m):
    if isinstance(m, SampledKWinnersBase):
        m.update_temperature()


class SampledKWinnersBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        percent_on,
        k_inference_factor=1.0,
        temperature=10.0,
        temperature_decay_rate=0.01,
        eval_temperature=1.0,
    ):
        super(SampledKWinnersBase, self).__init__()
        assert 0.0 < percent_on < 1.0
        assert 0.0 < percent_on * k_inference_factor < 1.0
        assert eval_temperature > 0.0

        self.percent_on = percent_on
        self.percent_on_inference = percent_on * k_inference_factor
        self.temperature = temperature
        self.temperature_decay_rate = temperature_decay_rate
        self.eval_temperature = eval_temperature
        self.learning_iterations = 0
        self.n = 0
        self.k = 0
        self.k_inference = 0
        # will be updated during forward passes
        self.total_entropy = -1.0

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)

    def extra_repr(self):
        return (
            "n={0}, percent_on={1}, temperature={2}, tempreature_decay_rate={3}, eval_temperature={4}".format(
                self.n, self.percent_on, self.temperature, self.temperature_decay_rate, self.eval_temperature
            )
        )

    def update_temperature(self):
        new_temp = self.temperature - self.temperature_decay_rate
        self.temperature = max(1.0, new_temp)

    def entropy(self):
        """Returns the current total entropy of this layer."""
        # TODO: update this
        return self.total_entropy

    def max_entropy(self):
        """Returns the maximum total entropy we can expect from this layer."""
        return max_entropy(self.n, int(self.n * self.percent_on))


class SampledKWinners(SampledKWinnersBase):
    def __init__(
        self,
        percent_on,
        k_inference_factor=1.5,
        temperature=10.0,
        eval_temperature=1.0,
        temperature_decay_rate=0.01,
        relu=False,
        inplace=False,
    ):

        super(SampledKWinners, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            temperature=temperature,
            eval_temperature=eval_temperature,
            temperature_decay_rate=temperature_decay_rate,
        )

        self.inplace = inplace
        self.relu = relu

    def forward(self, x):
        if self.n == 0:
            self.n = x.shape[-1]
            self.k = int(round(self.n * self.percent_on))
            self.k_inference = int(round(self.n * self.percent_on_inference))

        if self.training:
            x, entropy = F.sampled_kwinners(x, self.k, self.temperature, relu=self.relu)
        else:
            x, entropy = F.sampled_kwinners(x, self.k_inference, self.eval_temperature, relu=self.relu)
        self.total_entropy = entropy
        return x

    def extra_repr(self):
        s = super().extra_repr()
        if self.relu:
            s += ", relu=True"
        if self.inplace:
            s += ", inplace=True"
        return s


class SampledKWinners2d(SampledKWinnersBase):
    def __init__(
        self,
        percent_on=0.1,
        k_inference_factor=1.5,
        temperature=10.0,
        eval_temperature=1.0,
        temperature_decay_rate=0.01,
        relu=False,
        inplace=False
    ):

        super(SampledKWinners2d, self).__init__(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            temperature=temperature,
            eval_temperature=eval_temperature,
            temperature_decay_rate=temperature_decay_rate
        )

        self.inplace = inplace
        self.relu = relu


    def forward(self, x):
        if self.n == 0:
            self.n = np.prod(x.shape[1:])
            self.k = int(round(self.n * self.percent_on))
            self.k_inference = int(round(self.n * self.percent_on_inference))

        if self.training:
            x, entropy = F.sampled_kwinners2d(x, self.k, temperature=self.temperature, relu=self.relu, inplace=self.inplace)
        else:
            x, entropy = F.sampled_kwinners2d(x, self.k_inference, temperature=self.temperature, relu=self.relu, inplace=self.inplace)
        self.total_entropy = entropy
        return x

    def entropy(self):
        return self.total_entropy

    def extra_repr(self):
        s = super().extra_repr()
        if self.relu:
            s += ", relu=True"
        if self.inplace:
            s += ", inplace=True"
        return s
