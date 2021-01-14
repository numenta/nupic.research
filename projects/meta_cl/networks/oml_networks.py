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

import torch.nn as nn

from nupic.torch.modules.k_winners import KWinners


class KWinnerOMLNetwork(nn.Module):
    """
    This reproduces the network from the `OML`_ repository but with a kwinner
    layer on the penultimate layer.

    .. OML: https://github.com/khurramjaved96/mrcl

    With `num_classes=963`, it uses 5,172,675 weights in total.
    """

    def __init__(
        self,
        num_classes,
        percent_on=0.25,
        boost_strength_factor=0.9995
    ):

        super().__init__()

        self.representation = nn.Sequential(
            *self.conv_block(1, 256, 3, 2, 0),
            *self.conv_block(256, 256, 3, 1, 0),
            *self.conv_block(256, 256, 3, 2, 0),
            *self.conv_block(256, 256, 3, 1, 0),
            *self.conv_block(256, 256, 3, 2, 0),
            *self.conv_block(256, 256, 3, 2, 0),
            nn.Flatten(),
        )
        self.adaptation = nn.Sequential(
            KWinners(
                n=2304,
                percent_on=0.25,
                k_inference_factor=1.0,
                boost_strength=1.0,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=1000,
            ),
            nn.Linear(2304, num_classes),
        )

        # apply Kaiming initialization
        for param in self.parameters():
            if param.ndim > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    @classmethod
    def conv_block(cls, in_channels, out_channels, kernel_size, stride, padding):
        return [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.ReLU(),
        ]

    def forward(self, x):
        x = self.representation(x)
        x = self.adaptation(x)
        return x
