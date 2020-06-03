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

import numpy as np
from torch import nn

from nupic.research.frameworks.continuous_learning.dendrite_layers import DendriteLayer
from nupic.research.frameworks.pytorch.modules import KWinners2dLocal
from nupic.research.frameworks.pytorch.modules.consolidated_sparse_weights import (
    ConsolidatedSparseWeights,
    ConsolidatedSparseWeights2D,
)
from nupic.torch.modules import (
    Flatten,
    KWinners,
    KWinners2d,
    SparseWeights,
    SparseWeights2d,
)

# from nupic.research.frameworks.pytorch.models.le_sparse_net import (
#     add_sparse_cnn_layer,
#     add_sparse_linear_layer,
# )


def add_sparse_cnn_layer(
    network,
    suffix,
    in_channels,
    out_channels,
    use_batch_norm,
    weight_sparsity,
    percent_on,
    k_inference_factor,
    boost_strength,
    boost_strength_factor,
    duty_cycle_period,
    activation_fct_before_max_pool,
    use_kwinners_local,
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
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor:
        boost strength is multiplied by this factor after each epoch
    :param activation_fct_before_max_pool:
        If true ReLU/K-winners will be placed before the max_pool step
    :param use_kwinners_local:
        Whether or not to choose the k-winners 2d locally only across the
        channels instead of the whole input
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
        kwinner_class = KWinners2dLocal if use_kwinners_local else KWinners2d
        kwinner = kwinner_class(
            channels=out_channels,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
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
    boost_strength,
    boost_strength_factor,
    duty_cycle_period,
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
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor:
        boost strength is multiplied by this factor after each epoch
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
            KWinners(
                n=linear_n,
                percent_on=percent_on,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            ),
        )
    else:
        network.add_module("linear{}_relu".format(suffix), nn.ReLU())

    if dropout > 0.0:
        network.add_module("linear{}_dropout".format(suffix), nn.Dropout(dropout))


def add_sparse_dendrite_layer(
    network,
    suffix,
    in_dim,
    out_dim,
    dendrites_per_neuron,
    use_batch_norm=False,
    weight_sparsity=0.2,
    percent_on=0.1,
    k_inference_factor=1,
    boost_strength=1.5,
    boost_strength_factor=0.9,
    duty_cycle_period=1000
):

    dendrite_layer = DendriteLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        dendrites_per_neuron=dendrites_per_neuron,
        weight_sparsity=weight_sparsity,
    )

    network.add_module(
        "dendrites{}".format(suffix),
        dendrite_layer)

    if use_batch_norm:
        network.add_module("dendrites{}_bn".format(suffix),
                           nn.BatchNorm1d(out_dim, affine=False))

    network.add_module(
        "linear{}_kwinners".format(suffix),
        KWinners(
            n=out_dim,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        ),
    )


class LeSparseNet(nn.Sequential):
    """
    Experimental; modification of 'nupic.research.frameworks.pytorch.le_sparse_net'
    A generic LeNet style sparse CNN network as described in `How Can We Be So Dense?`_
    paper. The network has a set of CNN blocks, followed by a set of linear blocks,
    followed by a linear output layer.

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
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param use_kwinners_local:
        Whether or not to choose the k-winners 2d locally only across the
        channels instead of the whole input
    """

    def __init__(self,
                 input_shape=(1, 32, 32),
                 cnn_out_channels=(64, 64),
                 cnn_activity_percent_on=(0.1, 0.1),
                 cnn_weight_percent_on=(1.0, 1.0),
                 linear_n=(1000,),
                 linear_activity_percent_on=(0.1,),
                 linear_weight_percent_on=(0.4,),
                 use_dendrites=False,
                 dendrites_per_cell=5,
                 num_classes=10,
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 activation_fct_before_max_pool=False,
                 consolidated_sparse_weights=False,
                 use_kwinners_local=False,
                 use_softmax=True,
                 ):
        super(LeSparseNet, self).__init__()
        # Add CNN Layers
        current_input_shape = input_shape
        cnn_layers = len(cnn_out_channels)
        self.dpc = dendrites_per_cell
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
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
                activation_fct_before_max_pool=activation_fct_before_max_pool,
                use_kwinners_local=use_kwinners_local,
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
            if use_dendrites and i==0:
                add_sparse_dendrite_layer(
                    network=self,
                    suffix=i + 1,
                    in_dim=input_size,
                    out_dim=linear_n[i],
                    dendrites_per_neuron=self.dpc,
                    use_batch_norm=use_batch_norm,
                    weight_sparsity=linear_weight_percent_on[i],
                    percent_on=linear_activity_percent_on[i],
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                    duty_cycle_period=duty_cycle_period,
                )
            else:
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
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                    duty_cycle_period=duty_cycle_period,
                    consolidated_sparse_weights=consolidated_sparse_weights,
                )
            input_size = linear_n[i]

        if use_softmax:
            self.add_module("softmax", nn.LogSoftmax(dim=1))
