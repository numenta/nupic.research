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
import torch
from torch import nn

from nupic.research.frameworks.dendrites import (
    DendriteSegments,
    DendriticAbsoluteMaxGate1d,
    GatingDendriticLayer,
)
from nupic.research.frameworks.pytorch.model_utils import filter_params


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
                 dendrite_sparsity=0.50):

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


class ANMLDendriticNetwork(nn.Module):
    """
    Prototype of a dendritic network, based closely on `ANML`_. (The conv + max-pool
    layers are identical).

    .. _ANML: nupic.research/projects/meta_cl/networks/anml_networks.py

    Input shape expected is 3 x 28 x 28 (C x W x H) but an adaptive average pooling
    layer helps accept other heights and width.

    With default parameters and `num_classes-963`, it uses 5,118,291 weights-on
    out of a total of 8,345,095 weights. In comparison, ANML uses 5,963,139 weights
    and OML uses 5,172,675 weights. Thus, these is still room to add more.
    """

    def __init__(self, num_classes,
                 num_segments=20,
                 dim_context=100,
                 dendrite_sparsity=0.70,
                 dendrite_bias=None):

        super().__init__()

        self.segments = DendriteSegments(
            num_units=2304,
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )
        self.classifier = nn.Linear(2304, num_classes)

        self.prediction = nn.Sequential(
            *self.conv_block(3, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Flatten(),
        )

        self.modulation = nn.Sequential(
            *self.conv_block(3, 112, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(112, 112, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(112, 112, 3, 1, 0),
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Flatten(),
            nn.Linear(1008, dim_context)
        )

        self.dendritic_gate = DendriticAbsoluteMaxGate1d()

        # Apply Kaiming initialization
        self.reset_params()

    def reset_params(self):
        # Apply Kaiming initialization to Linear and Conv2d params
        named_params = filter_params(self, include_modules=[nn.Linear, nn.Conv2d])
        for _, param in named_params.items():
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
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(),
        ]

    def apply_dendrites(self, y, dendrite_activations):
        """
        Apply dendrites as a gating mechanism.
        """
        return self.dendritic_gate(y, dendrite_activations).values

    def forward(self, x):

        mod = self.modulation(x)
        pred = self.prediction(x)

        dendrite_activations = self.segments(mod)
        out = self.apply_dendrites(pred, dendrite_activations)

        out = self.classifier(out)

        return out


class CloserToANMLDendriticNetwork(nn.Module):
    """
    An iteration of a dendritic network, based closely on `ANML`_. (The conv + max-pool
    layers are identical). It's similar to ANMLDendriticNetwork, but does not include a
    linear layer atop the modulation network. This makes it more similar to ANML's
    original network. In fact, with `num_segments=1, dendrite_sparsity=0`, they should
    be identical.

    .. _ANML: nupic.research/projects/meta_cl/networks/anml_networks.py

    Input shape expected is 3 x 28 x 28 (C x W x H) but an adaptive average pooling
    layer helps accept other heights and width.

    With default parameters and `num_classes-963`, it uses 5,132,832 weights-on
    out of a total of 13,960,323 weights. In comparison, ANML uses 5,963,139 weights
    and OML uses 5,172,675 weights. Thus, these is still room to add more.

    The default parameters are chosen so that the number of on weights for this network
    is close to that of ANMLDendriticNetwork.
    """

    def __init__(self, num_classes,
                 num_segments=10,
                 dendrite_sparsity=0.856,
                 dendrite_bias=None):

        super().__init__()

        self.segments = DendriteSegments(
            num_units=2304,
            num_segments=num_segments,
            dim_context=1008,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )
        self.classifier = nn.Linear(2304, num_classes)

        self.prediction = nn.Sequential(
            *self.conv_block(3, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Flatten(),
        )

        self.modulation = nn.Sequential(
            *self.conv_block(3, 112, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(112, 112, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(112, 112, 3, 1, 0),
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Flatten(),
        )

        self.dendritic_gate = DendriticAbsoluteMaxGate1d()

        # Apply Kaiming initialization
        self.reset_params()

    def reset_params(self):
        # Apply Kaiming initialization to Linear and Conv2d params
        named_params = filter_params(self, include_modules=[nn.Linear, nn.Conv2d])
        for _, param in named_params.items():
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
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(),
        ]

    def apply_dendrites(self, y, dendrite_activations):
        """
        Apply dendrites as a gating mechanism.
        """
        return self.dendritic_gate(y, dendrite_activations).values

    def forward(self, x):
        mod = self.modulation(x)
        pred = self.prediction(x)

        dendrite_activations = self.segments(mod)
        out = self.apply_dendrites(pred, dendrite_activations)

        out = self.classifier(out)

        return out
