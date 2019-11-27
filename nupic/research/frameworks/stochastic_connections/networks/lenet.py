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

from torch import nn

from nupic.torch.modules import Flatten

__all__ = [
    "LeNet",
    "gsc_lenet",
    "mnist_lenet",
]


class LeNet(nn.Sequential):

    def __init__(self, input_size, num_classes,
                 cnn_out_channels=(64, 64),
                 kernel_size=5,
                 linear_units=1000,
                 maxpool_stride=2):
        feature_map_sidelength = (
            (((input_size[1] - kernel_size + 1) / maxpool_stride)
             - kernel_size + 1) / maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        super().__init__(OrderedDict([

            # -------------
            # Conv Block
            # -------------

            ("cnn1", nn.Conv2d(input_size[0],
                               cnn_out_channels[0],
                               kernel_size)),
            ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0],
                                       affine=False)),
            ("cnn1_relu", nn.ReLU()),

            # -------------
            # Conv Block
            # -------------

            ("cnn2", nn.Conv2d(cnn_out_channels[0],
                               cnn_out_channels[1],
                               kernel_size)),
            ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[1],
                                       affine=False)),
            ("cnn2_relu", nn.ReLU()),
            ("flatten", Flatten()),

            # -------------
            # Linear Block
            # -------------

            ("fc1", nn.Linear(
                (feature_map_sidelength**2) * cnn_out_channels[1],
                linear_units)),
            ("fc1_bn", nn.BatchNorm1d(linear_units, affine=False)),
            ("fc1_relu", nn.ReLU()),
            ("fc1_dropout", nn.Dropout(0.5)),

            # -------------
            # Output Layer
            # -------------

            ("fc2", nn.Linear(linear_units,
                              num_classes)),

        ]))


def gsc_lenet(input_size=(1, 32, 32), num_classes=12, **config):
    return LeNet(input_size=input_size, num_classes=num_classes, **config)


def mnist_lenet(input_size=(1, 28, 28), num_classes=10,
                cnn_out_channels=(32, 64), linear_units=700, **config):
    return LeNet(input_size=input_size, num_classes=num_classes,
                 cnn_out_channels=cnn_out_channels, linear_units=linear_units,
                 **config)
