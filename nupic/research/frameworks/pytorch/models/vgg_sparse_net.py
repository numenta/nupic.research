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
import math

from torch import nn

from nupic.torch.modules import (
    Flatten,
    KWinners,
    KWinners2d,
    SparseWeights,
    SparseWeights2d,
)


class VGGSparseNet(nn.Sequential):
    """
    A modified VGG style network that can be used to create different
    configurations of both sparse or dense VGG models as described in
    `How Can We Be So Dense?` paper

    :param input_shape: Shape of the input image. (C, H, W)
    :param block_sizes: number of CNN layers in each block
    :param cnn_out_channels: out_channels in each layer of this block
    :param cnn_kernel_sizes: kernel_size in each layer of this block
    :param cnn_weight_sparsity: weight sparsity of each layer of this block
    :param cnn_percent_on: percent_on in each layer of this block
    :param linear_units: Number of units in the linear layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param use_max_pooling: Whether or not to use MaxPool2d
    :param num_classes:  Number of output classes (10 for CIFAR10)
    """

    def __init__(self,
                 input_shape,
                 block_sizes,
                 cnn_out_channels,
                 cnn_kernel_sizes,
                 cnn_weight_sparsity,
                 cnn_percent_on,
                 linear_units,
                 linear_weight_sparsity,
                 linear_percent_on,
                 k_inference_factor,
                 boost_strength,
                 boost_strength_factor,
                 use_max_pooling,
                 num_classes,
                 ):
        super(VGGSparseNet, self).__init__()
        in_channels, h, w = input_shape
        output_size = h * w
        output_units = output_size * in_channels
        for l, block_size in enumerate(block_sizes):
            for b in range(block_size):
                self._add_cnn_layer(
                    index_str=str(l) + "_" + str(b),
                    in_channels=in_channels,
                    out_channels=cnn_out_channels[l],
                    kernel_size=cnn_kernel_sizes[l],
                    percent_on=cnn_percent_on[l],
                    weight_sparsity=cnn_weight_sparsity[l],
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                    add_pooling=b == block_size - 1,
                    use_max_pooling=use_max_pooling,
                )
                in_channels = cnn_out_channels[l]
            output_size = int(output_size / 4)
            output_units = output_size * in_channels

        # Flatten CNN output before passing to linear layer
        self.add_module("flatten", Flatten())

        # Linear layer
        input_size = output_units
        for l, linear_n in enumerate(linear_units):
            linear = nn.Linear(input_size, linear_n)
            if linear_weight_sparsity[l] < 1.0:
                self.add_module(
                    "linear_" + str(l),
                    SparseWeights(linear, linear_weight_sparsity[l]),
                )
            else:
                self.add_module("linear_" + str(l), linear)

            if linear_percent_on[l] < 1.0:
                self.add_module(
                    "kwinners_linear_" + str(l),
                    KWinners(
                        n=linear_n,
                        percent_on=linear_percent_on[l],
                        k_inference_factor=k_inference_factor,
                        boost_strength=boost_strength,
                        boost_strength_factor=boost_strength_factor,
                    ),
                )
            else:
                self.add_module("Linear_ReLU_" + str(l), nn.ReLU())

            input_size = linear_n

        # Output layer
        self.add_module("output", nn.Linear(input_size, num_classes))

        self._initialize_weights()

    def _add_cnn_layer(
        self,
        index_str,
        in_channels,
        out_channels,
        kernel_size,
        percent_on,
        weight_sparsity,
        k_inference_factor,
        boost_strength,
        boost_strength_factor,
        add_pooling,
        use_max_pooling,
    ):
        """Add a single CNN layer to our modules."""
        # Add CNN layer
        if kernel_size == 3:
            padding = 1
        else:
            padding = 2

        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        if weight_sparsity < 1.0:
            conv2d = SparseWeights2d(conv2d, weight_sparsity=weight_sparsity)
        self.add_module("cnn_" + index_str, conv2d)

        self.add_module("bn_" + index_str, nn.BatchNorm2d(out_channels)),

        if add_pooling:
            if use_max_pooling:
                self.add_module(
                    "maxpool_" + index_str, nn.MaxPool2d(kernel_size=2, stride=2)
                )
            else:
                self.add_module(
                    "avgpool_" + index_str, nn.AvgPool2d(kernel_size=2, stride=2)
                )

        if percent_on < 1.0:
            self.add_module(
                "kwinners_2d_" + index_str,
                KWinners2d(
                    percent_on=percent_on,
                    channels=out_channels,
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                ),
            )
        else:
            self.add_module("ReLU_" + index_str, nn.ReLU(inplace=True))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg19_sparse_net():
    """
    VGG-19 Sparse network configured for CIFAR-10 dataset as described in
    `How Can We Be So Dense?` paper
    """

    return VGGSparseNet(
        block_sizes=(2, 2, 4, 4, 4),
        input_shape=(3, 32, 32),
        cnn_out_channels=(64, 128, 256, 512, 512),
        cnn_kernel_sizes=(3, 3, 3, 3, 3),
        linear_weight_sparsity=(),
        linear_percent_on=(),
        cnn_percent_on=[0.25, 0.25, 0.25, 0.25, 0.25],
        cnn_weight_sparsity=[1.0, 0.5, 0.5, 0.5, 0.5],
        k_inference_factor=1.0,
        boost_strength=1.5,
        boost_strength_factor=0.85,
        use_max_pooling=True,
        num_classes=10,
    )


def vgg19_dense_net():
    """
    VGG-19 Dense network configured for CIFAR-10 dataset as described in
    `How Can We Be So Dense?` paper
    """

    return VGGSparseNet(
        block_sizes=(2, 2, 4, 4, 4),
        input_shape=(3, 32, 32),
        cnn_out_channels=(64, 128, 256, 512, 512),
        cnn_kernel_sizes=(3, 3, 3, 3, 3),
        linear_weight_sparsity=(),
        linear_percent_on=(),
        cnn_percent_on=[1.0, 1.0, 1.0, 1.0, 1.0],
        cnn_weight_sparsity=[1.0, 1.0, 1.0, 1.0, 1.0],
        k_inference_factor=1.0,
        boost_strength=1.5,
        boost_strength_factor=0.85,
        use_max_pooling=True,
        num_classes=10,
    )
