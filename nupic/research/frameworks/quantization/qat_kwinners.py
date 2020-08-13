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

import numpy as np

import nupic.torch.functions as F
from nupic.torch.modules import KWinners, KWinners2d

__all__ = [
    "QATKWinners",
    "QATKWinners2d",
    "QATKWINNER_MODULE_MAPPING"
]


class QATKWinners(KWinners):
    """
    Similar to nupic.torch.modules.KWinners, with FakeQuantize modules
    initialized to default.

    Attributes:
        activation_post_process: fake quant module for output activation
    """
    _FLOAT_MODULE = KWinners

    def __init__(self, n, percent_on, k_inference_factor, boost_strength_factor,
                 duty_cycle_period, qconfig):

        super().__init__(
            n=n,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, x):
        if self.training:
            x = F.KWinners.apply(x, self.duty_cycle, self.k, self.boost_strength)
            self.update_duty_cycle(x)
        else:
            x = F.KWinners.apply(x, self.duty_cycle, self.k_inference,
                                 self.boost_strength)

        return self.activation_post_process(x)

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, + cls.__name__ + \
            ".from_float only works for " + cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, "qconfig"), \
                "Input float module must have qconfig defined"
            assert mod.qconfig, "Input float module must have a valid qconfig"
            qconfig = mod.qconfig

        qat_kwinners = cls(
            n=mod.n,
            percent_on=mod.percent_on,
            k_inference_factor=mod.k_inference_factor,
            boost_strength_factor=mod.boost_strength_factor,
            duty_cycle_period=mod.duty_cycle_period,
            qconfig=qconfig
        )
        qat_kwinners.boost_strength = mod.boost_strength
        qat_kwinners.duty_cycle = mod.duty_cycle
        return qat_kwinners


class QATKWinners2d(KWinners2d):
    """
    Similar to nupic.torch.modules.KWinners, with FakeQuantize modules
    initialized to default.

    Attributes:
        activation_post_process: fake quant module for output activation
    """
    _FLOAT_MODULE = KWinners2d

    def __init__(self, channels, percent_on, k_inference_factor,
                 boost_strength_factor, duty_cycle_period, local, qconfig):

        super().__init__(
            channels=channels,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
            local=local
        )

        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, x):
        if self.n == 0:
            self.n = np.prod(x.shape[1:])
            if not self.local:
                self.k = int(round(self.n * self.percent_on))
                self.k_inference = int(round(self.n * self.percent_on_inference))

        if self.training:
            x = self.kwinner_function(x, self.duty_cycle, self.k, self.boost_strength)
            self.update_duty_cycle(x)
        else:
            x = self.kwinner_function(x, self.duty_cycle, self.k_inference,
                                      self.boost_strength)

        return self.activation_post_process(x)

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, + cls.__name__ + \
            ".from_float only works for " + cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, "qconfig"), \
                "Input float module must have qconfig defined"
            assert mod.qconfig, "Input float module must have a valid qconfig"
            qconfig = mod.qconfig

        qat_kwinners = cls(
            channels=mod.channels,
            percent_on=mod.percent_on,
            k_inference_factor=mod.k_inference_factor,
            boost_strength_factor=mod.boost_strength_factor,
            duty_cycle_period=mod.duty_cycle_period,
            local=mod.local,
            qconfig=qconfig
        )
        qat_kwinners.boost_strength = mod.boost_strength
        qat_kwinners.duty_cycle = mod.duty_cycle
        return qat_kwinners


QATKWINNER_MODULE_MAPPING = {
    KWinners: QATKWinners,
    KWinners2d: QATKWinners2d
}
