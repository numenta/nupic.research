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

from torch import nn

from nupic.research.frameworks.dendrites.dendritic_layers import GatingDendriticLayer


class DendriticNetwork(nn.Module):
    """
    Prototype of a dendritic network, based on ANML.

    It is composed of two parallel networks, one for prediction, and
    one for modulation. Each network has 3 conv layers with 256 channels
    each, followed by an adaptive average pool that reduces it to a 256x1x1.

    The output of the prediction and of the modulation are fed into a
    sparse linear gating layer as input and context respectively. The
    output of the gating layer are the logits used to calculate the loss function.

    With default parameters and `num_classes-963`, it uses 2,933,599 weights-on
    out of a total of 3,601,631 weights.
    """

    def __init__(self, num_classes,
                 num_segments=10,
                 dim_context=100,
                 module_sparsity=0.75,
                 dendrite_sparsity=0.50,
                 **kwargs):

        super().__init__()

        self.gating_layer = GatingDendriticLayer(  # <- linear + "den. segs"
            nn.Linear(256, num_classes),
            num_segments,
            dim_context,
            module_sparsity,  # % of weights that are zero
            dendrite_sparsity,  # % of dendrites that are zero
            dendrite_bias=None
        )

        self.prediction = nn.Sequential(
            *self.conv_block(1, 256, 3, 2, 0),
            *self.conv_block(256, 256, 3, 1, 0),
            *self.conv_block(256, 256, 3, 2, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.modulation = nn.Sequential(
            *self.conv_block(1, 256, 3, 2, 0),
            *self.conv_block(256, 256, 3, 1, 0),
            *self.conv_block(256, 256, 3, 2, 0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, dim_context)
        )

        # apply Kaiming initialization
        self.reset_params()

    def reset_params(self):
        # apply Kaiming initialization
        for param in self.prediction.parameters():
            if param.ndim > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

        for param in self.modulation.parameters():
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
        mod = self.modulation(x)
        pred = self.prediction(x)
        out = self.gating_layer(pred, context=mod)
        return out
