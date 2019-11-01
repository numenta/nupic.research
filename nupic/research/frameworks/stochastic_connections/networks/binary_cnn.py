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

from nupic.research.frameworks.stochastic_connections.binary_layers import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
)
from nupic.torch.modules import Flatten

__all__ = [
    "BinaryCNN",
    "gsc_binary_cnn",
    "mnist_binary_cnn",
]


class BinaryCNN(nn.Sequential):

    def __init__(self, config=None):

        config = config or {}
        defaults = dict(
            input_size=(1, 32, 32),
            l0_strength=7e-4,
            l2_strength=0,
            droprate_init=0.5,
            learn_weight=True,
            random_weight=False,
            num_classes=12,
            cnn_out_channels=(64, 64),
            kernel_size=5,
            linear_units=1000,
            maxpool_stride=2,
        )
        new_defaults = {k: (config.get(k, None) or v) for k, v in defaults.items()}
        self.__dict__.update(new_defaults)

        feature_map_sidelength = (
            (((self.input_size[1] - self.kernel_size + 1) / self.maxpool_stride)
             - self.kernel_size + 1) / self.maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        l0_strengths = [self.l0_strength] * 4

        super().__init__(OrderedDict([

            # -------------
            # Conv Block
            # -------------

            ("cnn1", BinaryGatedConv2d(
                self.input_size[0],
                self.cnn_out_channels[0],
                self.kernel_size,
                droprate_init=self.droprate_init,
                l2_strength=self.l2_strength,
                l0_strength=l0_strengths[0],
                learn_weight=self.learn_weight,
                random_weight=self.random_weight)),
            ("cnn1_bn", nn.BatchNorm2d(
                self.cnn_out_channels[0],
                affine=False)),
            ("cnn1_maxpool", nn.MaxPool2d(
                self.maxpool_stride)),
            ("cnn1_relu", nn.ReLU()),

            # -------------
            # Conv Block
            # -------------

            ("cnn2", BinaryGatedConv2d(
                self.cnn_out_channels[0],
                self.cnn_out_channels[1],
                self.kernel_size,
                droprate_init=self.droprate_init,
                l2_strength=self.l2_strength,
                l0_strength=l0_strengths[1],
                learn_weight=self.learn_weight,
                random_weight=self.random_weight)),
            ("cnn2_bn", nn.BatchNorm2d(
                self.cnn_out_channels[0],
                affine=False)),
            ("cnn2_maxpool", nn.MaxPool2d(
                self.maxpool_stride)),
            ("cnn2_relu", nn.ReLU()),
            ("flatten", Flatten()),

            # -------------
            # Linear Block
            # -------------

            ("fc1", BinaryGatedLinear(
                (feature_map_sidelength**2) * self.cnn_out_channels[1],
                self.linear_units,
                droprate_init=self.droprate_init,
                l2_strength=self.l2_strength,
                l0_strength=l0_strengths[2],
                learn_weight=self.learn_weight,
                random_weight=self.random_weight)),
            ("fc1_bn", nn.BatchNorm1d(
                self.linear_units,
                affine=False)),
            ("fc1_relu", nn.ReLU()),

            # -------------
            # Output Layer
            # -------------

            ("fc2", BinaryGatedLinear(
                self.linear_units,
                self.num_classes,
                droprate_init=self.droprate_init,
                l2_strength=self.l2_strength,
                l0_strength=l0_strengths[3],
                learn_weight=self.learn_weight,
                random_weight=self.random_weight)),

        ]))


def gsc_binary_cnn(config):
    config["input_size"] = (1, 32, 32)
    return BinaryCNN(config)


def mnist_binary_cnn(config):
    config["input_size"] = (1, 28, 28)
    return BinaryCNN(config)


if __name__ == "__main__":
    config = dict()
    gsc_binary_cnn(config)
    mnist_binary_cnn(config)
