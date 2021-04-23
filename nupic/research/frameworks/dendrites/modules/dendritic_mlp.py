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

import numpy as np
import torch
from torch import nn

from nupic.research.frameworks.dendrites.modules.dendritic_layers import (
    AbsoluteMaxGatingDendriticLayer,
    OneSegmentDendriticLayer,
)
from nupic.torch.modules import KWinners, SparseWeights, rezero_weights


class DendriticMLP(nn.Module):
    """
    A simple but restricted MLP with two hidden layers of the same size. Each hidden
    layer contains units with dendrites. Dendrite segments receive context directly as
    input.  The class is used to experiment with different dendritic weight
    initializations and learning parameters

    :param input_size: size of the input to the network
    :param output_size: the number of units in the output layer
    :param hidden_sizes: the number of units in each hidden layer
    :param num_segments: the number of dendritic segments that each hidden unit has
    :param dim_context: the size of the context input to the network
    :param kw: whether to apply k-Winners to the outputs of each hidden layer
    :param kw_percent_on: percent of hidden units activated by K-winners.
    :param context_percent_on: percent of non-zero units in the context input.
    :param weight_sparsity: the sparsity level of dendritic weights (default 0.95)
    :param weight_init: the initialization applied to feed-forward weights; must be
                        either "kaiming" (for Kaiming Uniform) of "modified" (for
                        sparse Kaiming Uniform)
    :param dendrite_init: the initialization applied to dendritic weights; similar to
                          `weight_init`
    :param freeze_dendrites: whether to set `requires_grad=False` for all dendritic
                             weights so they don't train
    :param dendritic_layer_class: dendritic layer class to use for each hidden layer
    :param output_nonlinearity: nonlinearity to apply to output. 'None' of no
                                nonlinearity.

                    _____
                   |_____|    # classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # first linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(
            self, input_size, output_size, hidden_sizes, num_segments, dim_context,
            kw, kw_percent_on=0.05, context_percent_on=1.0, weight_sparsity=0.95,
            weight_init="modified", dendrite_init="modified", freeze_dendrites=False,
            dendritic_layer_class=AbsoluteMaxGatingDendriticLayer, output_nonlinearity=None,
            preprocess_module_type=None, preprocess_output_dim=128,
            preprocess_kw_percent_on=0.1, representation_module_type=None,
            representation_module_dims=(128, 128), return_dendrite_outs=False
    ):

        # Forward & dendritic weight initialization must be either "kaiming" or
        # "modified"
        assert weight_init in ("kaiming", "modified")
        assert dendrite_init in ("kaiming", "modified")
        assert preprocess_module_type in (None, "relu", "kw")
        assert kw_percent_on >= 0.0
        assert context_percent_on >= 0.0

        super().__init__()

        if num_segments == 1:
            # use optimized 1 segment class
            dendritic_layer_class = OneSegmentDendriticLayer

        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw
        self.kw_percent_on = kw_percent_on
        self.weight_sparsity = weight_sparsity
        self.output_nonlinearity = output_nonlinearity
        self.hardcode_dendrites = (dendrite_init == "hardcoded")

        self.representation_dim = input_size
        # representation module: learns context(Task) independent representation of input
        self.representation_module = self._create_representation_module(
            representation_module_type,
            representation_module_dims
        )

        self.context_representation_dim = dim_context
        # preprocess module: builds a representation of context + input representation (for input to dendrite segments)
        self.preprocess_module = self._create_preprocess_module(
            preprocess_module_type,
            preprocess_output_dim,
            preprocess_kw_percent_on
        )

        self._dendrite_layers = nn.ModuleList()
        self._dendrite_activations = nn.ModuleList()

        input_size = self.representation_dim
        for i in range(len(self.hidden_sizes)):
            curr_dend = dendritic_layer_class(
                module=nn.Linear(input_size, self.hidden_sizes[i], bias=True),
                num_segments=num_segments,
                dim_context=self.context_representation_dim,
                module_sparsity=self.weight_sparsity,
                dendrite_sparsity=0.0 if self.hardcode_dendrites else weight_sparsity,
            )

            if weight_init == "modified":
                # Scale weights to be sampled from the new initialization U(-h, h) where
                # h = sqrt(1 / (weight_density * previous_layer_percent_on))
                if i == 0:
                    # first hidden layer can't have kw input
                    self._init_sparse_weights(curr_dend, 0.0)
                else:
                    self._init_sparse_weights(
                        curr_dend,
                        1 - kw_percent_on if kw else 0.0
                    )

            if dendrite_init == "modified":
                self._init_sparse_dendrites(curr_dend, 1 - context_percent_on)

            if freeze_dendrites:
                # Dendritic weights will not be updated during backward pass
                for name, param in curr_dend.named_parameters():
                    if "segments" in name:
                        param.requires_grad = False

            if self.kw:
                curr_activation = KWinners(n=hidden_sizes[i],
                                           percent_on=kw_percent_on,
                                           k_inference_factor=1.0,
                                           boost_strength=0.0,
                                           boost_strength_factor=0.0)
            else:
                curr_activation = nn.ReLU()

            self._dendrite_layers.append(curr_dend)
            self._dendrite_activations.append(curr_activation)

            input_size = self.hidden_sizes[i]

        self._output_layer = nn.Sequential()
        output_linear = SparseWeights(module=nn.Linear(input_size, output_size),
                                      sparsity=weight_sparsity, allow_extremes=True)
        if weight_init == "modified":
            self._init_sparse_weights(output_linear, 1 - kw_percent_on if kw else 0.0)
        self._output_layer.add_module("output_linear", output_linear)

        if self.output_nonlinearity:
            self._output_layer.add_module("non_linearity", output_nonlinearity)

    def forward(self, x, context, return_sigmoid_values=False):
        if self.representation_module is not None:
            x = self.representation_module(x)
        if self.preprocess_module is not None:
            context = self.preprocess_module(torch.cat((x, context), dim=-1))

        if return_sigmoid_values:
            dendrite_sigmoids = []

        for layer, activation in zip(self._dendrite_layers, self._dendrite_activations):
            if return_sigmoid_values:
                x, dend_sig = layer(x, context, return_sigmoid_values=True)
                x = activation(x)
                dendrite_sigmoids.append(dend_sig)
            else:
                x = activation(layer(x, context))

        if return_sigmoid_values:
            return self._output_layer(x), torch.stack(dendrite_sigmoids)
        else:
            return self._output_layer(x)

    def _create_representation_module(self, module_type, dims):
        if module_type is None:
            return None
        representation_module = nn.Sequential()

        inp_dim = self.input_size
        for i in range(len(dims)):
            output_dim = dims[i]
            layer = SparseWeights(
                torch.nn.Linear(inp_dim,
                                output_dim,
                                bias=True),
                sparsity=self.weight_sparsity,
                allow_extremes=True
            )
            # network input is dense (no sparsity constraints)
            self._init_sparse_weights(layer, 0.0)

            if module_type == "relu":
                nonlinearity = nn.ReLU()
            else:
                raise NotImplementedError
            representation_module.add_module("linear_layer_{}".format(i), layer)
            representation_module.add_module("nonlinearity_{}".format(i), nonlinearity)
            inp_dim = output_dim

        self.representation_dim = inp_dim
        return representation_module


    def _create_preprocess_module(self, module_type, preprocess_output_dim, kw_percent_on):
        if module_type is None:
            return None

        preprocess_module = nn.Sequential()
        linear_layer = SparseWeights(
            torch.nn.Linear(self.dim_context + self.representation_dim,
                            preprocess_output_dim,
                            bias=True),
            sparsity=self.weight_sparsity,
            allow_extremes=True
        )
        self._init_sparse_weights(linear_layer, 0.0)

        if module_type == "relu":
            nonlinearity = nn.ReLU()
        else:
            nonlinearity = KWinners(
                n=preprocess_output_dim,
                percent_on=kw_percent_on,
                k_inference_factor=1.0,
                boost_strength=0.0,
                boost_strength_factor=0.0
            )
        preprocess_module.add_module("linear_layer", linear_layer)
        preprocess_module.add_module("nonlinearity", nonlinearity)

        self.context_representation_dim = preprocess_output_dim
        return preprocess_module

    # ------ Weight initialization functions ------
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

    @staticmethod
    def _init_sparse_dendrites(m, input_sparsity):
        """
        Modified Kaiming initialization for dendrites segments that consider input
        sparsity and dendritic weight sparsity.
        """
        # Assume `m` is an instance of `DendriticLayerBase`
        input_density = 1.0 - input_sparsity
        weight_density = 1.0 - m.segments.sparsity
        fan_in = m.dim_context
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.weights, -bound, bound)
        m.apply(rezero_weights)

    def hardcode_dendritic_weights(self, context_vectors, init):
        """
        Set up specific weights for each dendritic segment based on the value of init.

        if init == "overlapping":
            We hardcode the weights of dendrites such that each context selects 5% of
            hidden units to become active and form a subnetwork. Hidden units are
            sampled with replacement, hence subnetworks can overlap. Any context/task
            which does not use a particular hidden unit will cause it to turn off, as
            the unit's other segment(s) have -1 in all entries and will yield an
            extremely small dendritic activation.

        otherwise if init == "non_overlapping":
            We hardcode the weights of dendrites such that each unit recognizes a single
            random context vector. The first dendritic segment is initialized to contain
            positive weights from that context vector. The other segment(s) ensure that
            the unit is turned off for any other context - they contain negative weights
            for all other weights.

        :param context_vectors:
        :param init: a string "overlapping" or "non_overlapping"
        """
        for dendrite in self._dendrite_layers:
            self._hardcode_dendritic_weights(dendrite.weights, context_vectors, init)

    @staticmethod
    def _hardcode_dendritic_weights(dendrite_weights, context_vectors, init):
        assert self.preprocess_module is None
        squeeze = False
        if len(dendrite_weights.shape) == 2:
            # 1 segment dendrite, so add in a segment dimension
            squeeze = True
            original_weights = dendrite_weights
            dendrite_weights = dendrite_weights.unsqueeze(dim=1)

        num_units, num_segments, dim_context = dendrite_weights.size()
        num_contexts, _ = context_vectors.size()

        if init == "overlapping":
            new_dendritic_weights = -0.95 * torch.ones((num_units, num_segments,
                                                        dim_context))

            # The number of units to allocate to each context (with replacement)
            k = int(0.05 * num_units)

            # Keep track of the number of contexts for which each segment has already
            # been chosen; this is to not overwrite a previously hardcoded segment
            num_contexts_chosen = {i: 0 for i in range(num_units)}

            for c in range(num_contexts):

                # Pick k random units to be activated by the cth context
                selected_units = torch.randperm(num_units)[:k]
                for i in selected_units:
                    i = i.item()

                    # If num_segments other contexts have already selected unit i to
                    # become active, skip
                    segment_id = num_contexts_chosen[i]
                    if segment_id == num_segments:
                        continue

                    new_dendritic_weights[i, segment_id, :] = context_vectors[c, :]
                    num_contexts_chosen[i] += 1

        elif init == "non_overlapping":
            new_dendritic_weights = torch.zeros((num_units, num_segments, dim_context))

            for i in range(num_units):
                context_perm = context_vectors[torch.randperm(num_contexts), :]
                new_dendritic_weights[i, :, :] = 1.0 * (context_perm[0, :] > 0)
                new_dendritic_weights[i, 1:, :] = -1
                new_dendritic_weights[i, 1:, :] += new_dendritic_weights[i, 0, :]
                del context_perm

        else:
            raise Exception("Invalid dendritic weight hardcode choice")

        dendrite_weights.data = new_dendritic_weights

        if squeeze:
            dendrite_weights = dendrite_weights.squeeze(dim=1)
            # dendrite weights doesn't point to the dendrite weights tensor,
            # so expicitly assign the new values
            original_weights.data = dendrite_weights


if __name__ == '__main__':
    d = DendriticMLP(10, 5, (32, 32), num_segments=1, dim_context=10, context_percent_on=1.0, kw=True,
                     representation_module_type="relu", preprocess_module_type="relu")
    # contexts = torch.eye(10)
    # d.hardcode_dendritic_weights(contexts, init="non_overlapping")
    ret = d(torch.rand(32, 10), torch.rand(32, 10), return_sigmoid_values=True)
    a = 2
