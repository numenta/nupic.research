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
import torch.nn as nn

from nupic.research.frameworks.pytorch.models.le_sparse_net import (
    add_sparse_linear_layer,
)

__all__ = [
    "StandardMLP",
    "ModifiedInitStandardMLP",
    "SparseMLP",
]


class StandardMLP(nn.Module):

    def __init__(self, input_size, num_classes,
                 hidden_sizes=(100, 100)):
        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(int(np.prod(input_size)), hidden_sizes[0]),
            nn.ReLU()
        ]
        for idx in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class ModifiedInitStandardMLP(StandardMLP):
    """
    A standard MLP which differs only in feed-forward weight initialization: the bounds
    of the Uniform distribution used to initialization weights are

    +/- 1/sqrt(I x W x F)

    where I is the density of the input for a given layer, W is always 1.0 (since MLPs
    have dense weights), and F is fan-in. This only differs from Kaiming Uniform
    initialization by incorporating input density (I) and weight density (W). Biases
    are unaffected.
    """

    def __init__(self, input_size, num_classes, hidden_sizes):
        super().__init__(input_size, num_classes, hidden_sizes)

        # Modified Kaiming weight initialization which considers 1) the density of
        # the input vector and 2) the weight density in addition to the fan-in

        weight_density = 1.0
        input_flag = False
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):

                # Assume input is fully dense, but hidden layer activations are only
                # 50% dense due to ReLU
                input_density = 1.0 if not input_flag else 0.5
                input_flag = True

                _, fan_in = layer.weight.size()
                bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
                nn.init.uniform_(layer.weight, -bound, bound)


class SparseMLP(nn.Sequential):
    input_ = """
    A sparse MLP network which contains sparse linear hidden layers followed by a dense
    linear output layer. Each hidden layer contains a sparse linear layer, optional
    batch norm, and a k-winner
    :param input_size: Input dimension to the network.
    :param output_size: Output dimension of the network.
    :param linear_activity_percent_on: Percent of ON (non-zero) units
    :param linear_weight_percent_on: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param use_batch_norm: whether to use batch norm
    :param dropout: dropout value
    :param consolidated_sparse_weights: whether to use consolidated sparse weights
    :param hidden_sizes: hidden layer dimensions of MLP
    """

    def __init__(self, input_size,
                 output_size,
                 kw_percent_on=(0.1,),
                 weight_sparsity=(0.6,),
                 boost_strength=1.67,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.5,
                 use_batch_norm=True,
                 dropout=0.0,
                 consolidated_sparse_weights=False,
                 hidden_sizes=(100,)):
        super().__init__()
        self.add_module("flatten", nn.Flatten())
        # Add Sparse Linear layers
        for i in range(len(hidden_sizes)):
            add_sparse_linear_layer(
                network=self,
                suffix=i + 1,
                input_size=input_size,
                linear_n=hidden_sizes[i],
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                weight_sparsity=weight_sparsity[i],
                percent_on=kw_percent_on[i],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
                consolidated_sparse_weights=consolidated_sparse_weights,
            )
            input_size = hidden_sizes[i]
        self.add_module("output", nn.Linear(input_size, output_size))
