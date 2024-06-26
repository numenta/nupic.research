#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
from torch import nn

from nupic.research.frameworks.pytorch.modules.consolidated_sparse_weights import (
    ConsolidatedSparseWeights,
    ConsolidatedSparseWeights2D,
)
from nupic.torch.modules import Flatten, SparseWeights, SparseWeights2d

from .sampled_kwinners import SampledKWinners, SampledKWinners2d


def add_sparse_cnn_layer(
    network,
    suffix,
    in_channels,
    out_channels,
    use_batch_norm,
    weight_sparsity,
    percent_on,
    k_inference_factor,
    temperature,
    eval_temperature,
    temperature_decay_rate,
    activation_fct_before_max_pool,
    consolidated_sparse_weights,
):
    """Add sparse cnn layer to network.

    :param network: The network to add the sparse layer to
    :param suffix: Layer suffix. Used to name its components
    :param in_channels: input channels
    :param out_channels: output channels
    :param use_batch_norm: whether or not to use batch norm
    :param weight_sparsity: Pct of weights that are allowed to be non-zero
    :param percent_on: Pct of ON (non-zero) units
    :param k_inference_factor: During inference we increase percent_on by this factor
    :param temperature: temperature to use when computing softmax in SampledKWinners.
    :param eval_temperature:
        temperature to use when computing softmax during evaluation in SampledKWinners.
    :param temperature_decay_rate:
        amount to decrease the temperature after each epoch of training.
    :param activation_fct_before_max_pool:
        If true ReLU/K-winners will be placed before the max_pool step
    """
    cnn = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        padding=0,
        stride=1,
    )
    if 0 < weight_sparsity < 1.0:
        if consolidated_sparse_weights:
            sparse_cnn = ConsolidatedSparseWeights2D(cnn, weight_sparsity)
        else:
            sparse_cnn = SparseWeights2d(cnn, weight_sparsity)
        network.add_module("cnn{}_cnn".format(suffix), sparse_cnn)
    else:
        network.add_module("cnn{}_cnn".format(suffix), cnn)

    if use_batch_norm:
        bn = nn.BatchNorm2d(out_channels, affine=False)
        network.add_module("cnn{}_bn".format(suffix), bn)

    if not activation_fct_before_max_pool:
        maxpool = nn.MaxPool2d(kernel_size=2)
        network.add_module("cnn{}_maxpool".format(suffix), maxpool)

    if percent_on >= 1.0 or percent_on <= 0:
        network.add_module("cnn{}_relu".format(suffix), nn.ReLU())
    else:
        kwinner_class = SampledKWinners2d
        kwinner = kwinner_class(
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            temperature=temperature,
            eval_temperature=eval_temperature,
            temperature_decay_rate=temperature_decay_rate,
            relu=False
        )
        network.add_module("cnn{}_kwinner".format(suffix), kwinner)

    if activation_fct_before_max_pool:
        maxpool = nn.MaxPool2d(kernel_size=2)
        network.add_module("cnn{}_maxpool".format(suffix), maxpool)


def add_sparse_linear_layer(
    network,
    suffix,
    input_size,
    linear_n,
    dropout,
    use_batch_norm,
    weight_sparsity,
    percent_on,
    k_inference_factor,
    temperature,
    eval_temperature,
    temperature_decay_rate,
    consolidated_sparse_weights,
):
    """Add sparse linear layer to network.

    :param network: The network to add the sparse layer to
    :param suffix: Layer suffix. Used to name its components
    :param input_size: Input size
    :param linear_n: Number of units
    :param dropout: dropout value
    :param use_batch_norm: whether or not to use batch norm
    :param weight_sparsity: Pct of weights that are allowed to be non-zero
    :param percent_on: Pct of ON (non-zero) units
    :param k_inference_factor: During inference we increase percent_on by this factor
    :param temperature: temperature to use when computing softmax in SampledKWinners.
    :param eval_temperature:
        temperature to use when computing softmax during evaluation in SampledKWinners.
    :param temperature_decay_rate:
        amount to decrease the temperature after each epoch of training.

    """
    linear = nn.Linear(input_size, linear_n)
    if 0 < weight_sparsity < 1.0:
        if consolidated_sparse_weights:
            network.add_module(
                "linear{}".format(suffix),
                ConsolidatedSparseWeights(linear, weight_sparsity)
            )
        else:
            network.add_module(
                "linear{}".format(suffix), SparseWeights(linear, weight_sparsity)
            )
    else:
        network.add_module("linear{}_linear".format(suffix), linear)

    if use_batch_norm:
        network.add_module("linear{}_bn".format(suffix),
                           nn.BatchNorm1d(linear_n, affine=False))

    if 0 < percent_on < 1.0:
        network.add_module(
            "linear{}_kwinners".format(suffix),
            SampledKWinners(
                percent_on=percent_on,
                k_inference_factor=k_inference_factor,
                temperature=temperature,
                eval_temperature=eval_temperature,
                temperature_decay_rate=temperature_decay_rate,
                relu=False
            ),
        )
    else:
        network.add_module("linear{}_relu".format(suffix), nn.ReLU())

    if dropout > 0.0:
        network.add_module("linear{}_dropout".format(suffix), nn.Dropout(dropout))


class SampledKWinnerLeSparseNet(nn.Sequential):
    """
    A generic LeNet style sparse CNN network using Sampled KWinners
    as described in `How Can We Be So Dense?` paper. The network has a set of CNN
    blocks, followed by a set of linear blocks, followed by a linear output layer.

    Each CNN block contains a CNN layer, optional batch norm, k-winner or ReLU, followed
    by maxpool.  Each linear block contains a linear layer, optional batch norm, and a
    k-winner or ReLU.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param temperature: temperature to use when computing softmax in SampledKWinners.
    :param eval_temperature:
        temperature to use when computing softmax during evaluation in SampledKWinners.
    :param temperature_decay_rate:
        amount to decrease the temperature after each epoch of training.
    """

    def __init__(self,
                 input_shape=(1, 32, 32),
                 cnn_out_channels=(64, 64),
                 cnn_activity_percent_on=(0.1, 0.1),
                 cnn_weight_percent_on=(1.0, 1.0),
                 linear_n=(1000,),
                 linear_activity_percent_on=(0.1,),
                 linear_weight_percent_on=(0.4,),
                 num_classes=10,
                 temperature=10.0,
                 eval_temperature=1.0,
                 temperature_decay_rate=0.99,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 activation_fct_before_max_pool=False,
                 consolidated_sparse_weights=False,
                 use_softmax=True,
                 ):
        super(SampledKWinnerLeSparseNet, self).__init__()

        # Add CNN Layers
        current_input_shape = input_shape
        cnn_layers = len(cnn_out_channels)
        for i in range(cnn_layers):
            in_channels, height, width = current_input_shape

            # We only do consolidated weights for the second CNN layer
            csw = (i == 1) and consolidated_sparse_weights
            add_sparse_cnn_layer(
                network=self,
                suffix=i + 1,
                in_channels=in_channels,
                out_channels=cnn_out_channels[i],
                use_batch_norm=use_batch_norm,
                weight_sparsity=cnn_weight_percent_on[i],
                percent_on=cnn_activity_percent_on[i],
                k_inference_factor=k_inference_factor,
                temperature=temperature,
                eval_temperature=eval_temperature,
                temperature_decay_rate=temperature_decay_rate,
                activation_fct_before_max_pool=activation_fct_before_max_pool,
                consolidated_sparse_weights=csw
            )

            # Compute next layer input shape
            wout = (width - 5) + 1
            maxpool_width = wout // 2
            current_input_shape = (cnn_out_channels[i], maxpool_width, maxpool_width)

        # Flatten CNN output before passing to linear layer
        self.add_module("flatten", Flatten())

        # Add Linear layers
        input_size = np.prod(current_input_shape)
        for i in range(len(linear_n)):
            add_sparse_linear_layer(
                network=self,
                suffix=i + 1,
                input_size=input_size,
                linear_n=linear_n[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                weight_sparsity=linear_weight_percent_on[i],
                percent_on=linear_activity_percent_on[i],
                k_inference_factor=k_inference_factor,
                temperature=temperature,
                eval_temperature=eval_temperature,
                temperature_decay_rate=temperature_decay_rate,
                consolidated_sparse_weights=consolidated_sparse_weights,
            )
            input_size = linear_n[i]

        # Classifier
        self.add_module("output", nn.Linear(input_size, num_classes))
        if use_softmax:
            self.add_module("softmax", nn.LogSoftmax(dim=1))
