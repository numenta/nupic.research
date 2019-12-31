# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from collections import OrderedDict
from functools import partial

from torch import nn

from nupic.research.frameworks.backprop_structure.modules.binary_layers import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
)

__all__ = [
    "LeNetBackpropStructure",
    "gsc_lenet_backpropstructure",
    "mnist_lenet_backpropstructure",
]


class LeNetBackpropStructure(nn.Sequential):

    def __init__(self, input_size, num_classes,
                 l0_strength=7e-6,
                 l2_strength=0,
                 droprate_init=0.5,
                 learn_weight=True,
                 random_weight=True,
                 cnn_out_channels=(64, 64),
                 kernel_size=5,
                 linear_units=1000,
                 maxpool_stride=2,
                 bn_track_running_stats=True):
        feature_map_sidelength = (
            (((input_size[1] - kernel_size + 1) / maxpool_stride)
             - kernel_size + 1) / maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        common_params = dict(
            droprate_init=droprate_init,
            l2_strength=l2_strength,
            l0_strength=l0_strength,
            learn_weight=learn_weight,
            random_weight=random_weight
        )

        super().__init__(OrderedDict([

            # -------------
            # Conv Block
            # -------------

            ("cnn1", BinaryGatedConv2d(input_size[0],
                                       cnn_out_channels[0],
                                       kernel_size,
                                       **common_params)),
            ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn1_bn", nn.BatchNorm2d(
                cnn_out_channels[0],
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn1_relu", nn.ReLU(inplace=True)),

            # -------------
            # Conv Block
            # -------------

            ("cnn2", BinaryGatedConv2d(cnn_out_channels[0],
                                       cnn_out_channels[1],
                                       kernel_size,
                                       **common_params)),
            ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn2_bn", nn.BatchNorm2d(
                cnn_out_channels[1],
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn2_relu", nn.ReLU(inplace=True)),
            ("flatten", nn.Flatten()),

            # -------------
            # Linear Block
            # -------------

            ("fc1", BinaryGatedLinear(
                (feature_map_sidelength**2) * cnn_out_channels[1],
                linear_units,
                **common_params)),
            ("fc1_bn", nn.BatchNorm1d(
                linear_units,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("fc1_relu", nn.ReLU(inplace=True)),

            # -------------
            # Output Layer
            # -------------

            ("fc2", BinaryGatedLinear(linear_units,
                                      num_classes,
                                      **common_params)),

        ]))


gsc_lenet_backpropstructure = partial(
    LeNetBackpropStructure,
    input_size=(1, 32, 32),
    num_classes=12)

mnist_lenet_backpropstructure = partial(
    LeNetBackpropStructure,
    input_size=(1, 28, 28),
    num_classes=10,
    cnn_out_channels=(32, 64),
    linear_units=700)
