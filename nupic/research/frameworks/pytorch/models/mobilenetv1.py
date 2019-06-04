#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import torch.nn as nn

from nupic.torch.modules import Flatten, KWinners2d


def separable_convolution2d(
    in_channels, out_channels, kernel_size=(3, 3), stride=1, width_mult=1.0
):
    """Depth wise separable convolution 2D. This network block is used by
    MobileNet to factorize a standard convolution into a depth wise convolution
    and a 1x1 point wise convolution. The depth wise convolution applies a
    single filter for each input channel and the point wise applies 1x1
    convolution to combine the outputs of the depth wise convolution.

    See  https://arxiv.org/abs/1704.04861

    :param in_channels: Input channels
    :param out_channels: Output channels
    :param kernel_size: Kernel size to use, always 3x3 for mobilenet
    :param stride: Stride of the convolution
    :param width_mult: Width multiplier, used to thin the network
    """
    # Apply width multiplier (alpha)
    in_channels = int(in_channels * width_mult)
    out_channels = int(out_channels * width_mult)
    return nn.Sequential(
        # Depth wise convolution
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(True),
        # Point wise convolution
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


class MobileNetV1(nn.Module):
    """See https://arxiv.org/abs/1704.04861."""

    def __init__(self, num_classes=1001, width_mult=1.0):
        """Inspired by https://github.com/kuangliu/pytorch-
        cifar/blob/master/models/mobilenet.py.

        :param num_classes: Number of output classes (10 for CIFAR10)
        :param width_mult: Width multiplier, used to thin the network
        """
        super(MobileNetV1, self).__init__()

        # Check for CIFAR10
        if num_classes == 10:
            first_stride = 1
            avgpool_size = 2
        else:
            first_stride = 2
            avgpool_size = 7

        # First 3x3 convolution layer
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=int(32 * width_mult),
                kernel_size=3,
                stride=first_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(32 * width_mult)),
            nn.ReLU(True),
        )

        # Depthwise Separable Convolution layers
        self.deepwise = nn.Sequential(
            separable_convolution2d(
                in_channels=32, out_channels=64, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=64, out_channels=128, stride=2, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=128, out_channels=128, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=128, out_channels=256, stride=2, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=256, out_channels=256, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=256, out_channels=512, stride=2, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=512, out_channels=512, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=512, out_channels=512, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=512, out_channels=512, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=512, out_channels=512, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=512, out_channels=512, stride=1, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=512, out_channels=1024, stride=2, width_mult=width_mult
            ),
            separable_convolution2d(
                in_channels=1024, out_channels=1024, stride=1, width_mult=width_mult
            ),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AvgPool2d(avgpool_size),
            Flatten(),
            nn.Linear(in_features=int(1024 * width_mult), out_features=num_classes),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.deepwise(y)
        y = self.classifier(y)
        return y


def mobile_net_v1_sparse_depth(
    num_classes=1001,
    width_mult=1.0,
    percent_on=0.1,
    k_inference_factor=1.0,
    boost_strength=1.0,
    boost_strength_factor=1.0,
    duty_cycle_period=1000,
):
    """Create a MobileNetV1 network with sparse deep wise layers by replacing
    the Depth wise (3x3) convolution activation function from ReLU with
    k-winners.

    :param num_classes:
      Number of output classes (10 for CIFAR10)
    :type num_classes: int

    :param width_mult:
      Width multiplier, used to thin the network
    :type width_mult: float

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param kInferenceFactor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * kInferenceFactor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int

    :return: Depth wise Sparse MoblineNetV1 model
    """
    model = MobileNetV1(num_classes=num_classes, width_mult=width_mult)
    # Replace Deep wise ReLU (3rd layer) with k-winners
    for block in model.deepwise:
        # Get number of features from previous BatchNorm2d layer
        channels = block[1].num_features
        block[2] = KWinners2d(
            channels,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )
    return model


def mobile_net_v1_sparse_point(
    num_classes=1001,
    width_mult=1.0,
    percent_on=0.1,
    k_inference_factor=1.0,
    boost_strength=1.0,
    boost_strength_factor=1.0,
    duty_cycle_period=1000,
):
    """Create a MobileNetV1 network with sparse point wise layers by replacing
    the Point wise (1x1) convolution activation function from ReLU with
    k-winners.

    :param num_classes:
      Number of output classes (10 for CIFAR10)
    :type num_classes: int

    :param width_mult:
      Width multiplier, used to thin the network
    :type width_mult: float

    :param percent_on:
      The activity of the top k = percent_on * number of input units will be
      allowed to remain, the rest are set to zero.
    :type percent_on: float

    :param kInferenceFactor:
      During inference (training=False) we increase percent_on by this factor.
      percent_on * kInferenceFactor must be strictly less than 1.0, ideally much
      lower than 1.0
    :type kInferenceFactor: float

    :param boostStrength:
      boost strength (0.0 implies no boosting).
    :type boostStrength: float

    :param boostStrengthFactor:
      Boost strength factor to use [0..1]
    :type boostStrengthFactor: float

    :param dutyCyclePeriod:
      The period used to calculate duty cycles
    :type dutyCyclePeriod: int

    :return: Point wise Sparse MoblineNetV1 model
    """
    model = MobileNetV1(num_classes=num_classes, width_mult=width_mult)

    # Replace Point wise ReLU (6th layer) with k-winners
    for block in model.deepwise:
        # Get number of features from previous BatchNorm2d layer
        channels = block[4].num_features
        block[5] = KWinners2d(
            channels,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

    return model
