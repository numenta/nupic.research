# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import abc

import numpy as np
import torch.nn as nn

from nupic.torch.modules import KWinners, SparseWeights, rezero_weights


class SP(nn.Module):
    """
    A simple spatial pooler like network to be used as a context vector.

    :param input_size: size of the input to the network
    :param output_size: the number of units in the output layer
    :param kw_percent_on: percent of hidden units activated by K-winners. If 0, use ReLU
    :param boost_strength:
    :param weight_sparsity: the sparsity level of feed-forward weights.
    :param duty_cycle_period:
    """

    def __init__(
        self, input_size, output_size, kw_percent_on=0.05, boost_strength=0.0,
        weight_sparsity=0.95, duty_cycle_period=1000,
    ):
        super().__init__()

        self.context_linear = SparseWeights(nn.Linear(input_size, output_size),
                                            sparsity=weight_sparsity,
                                            allow_extremes=True)
        self.context_kw = KWinners(n=output_size, percent_on=kw_percent_on,
                                   boost_strength=boost_strength,
                                   duty_cycle_period=duty_cycle_period,
                                   k_inference_factor=1.0)

    def forward(self, x):
        return self.context_kw(self.context_linear(x))

    @staticmethod
    def _init_sparse_weights(m, input_sparsity):
        """
        Modified Kaiming weight initialization that considers input sparsity and weight
        sparsity.
        """
        input_density = 1.0 - input_sparsity
        weight_density = 1.0 - m.sparsity
        _, fan_in = m.module.weight.size()
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.module.weight, -bound, bound)
        m.apply(rezero_weights)


class SpatialPoolerContext(metaclass=abc.ABCMeta):
    """
    Use an untrained SP to output context vectors.
    """

    def setup_experiment(self, config):
        context_model_args = config.get("context_model_args")
        context_out = context_model_args.get("output_size")
        dim_context = config["model_args"].get("dim_context")

        assert dim_context == context_out, \
            ("`dim_context` must match context network output")

        super().setup_experiment(config)

        self.context_network = SP(**context_model_args)
        self.context_network.to(self.device)

        print(self.context_network)

        # We're not going to train this network, but we need ta accumulate k-winner
        # duty cycle if boost strength is positive
        self.context_network.eval()
        self.context_network.context_kw.train()

        self.infer_context_fn = self.context_network
        self.train_context_fn = self.context_network
