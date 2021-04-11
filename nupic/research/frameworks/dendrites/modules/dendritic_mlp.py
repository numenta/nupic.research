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

from nupic.torch.modules import KWinners, SparseWeights, rezero_weights

from .dendritic_layers import AbsoluteMaxGatingDendriticLayer


class DendriticMLP(nn.Module):
    """
    A simple but restricted MLP with two hidden layers of the same size. Each hidden
    layer contains units with dendrites. Dendrite segments receive context directly as
    input.  The class is used to experiment with different dendritic weight
    initializations and learning parameters

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

    def __init__(self, input_size, output_size, hidden_size, num_segments, dim_context,
                 weight_init, dendrite_init, kw,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        # Forward weight initialization must of one of "kaiming" or "modified" (i.e.,
        # modified sparse Kaiming initialization)
        assert weight_init in ("kaiming", "modified")

        # Forward weight initialization must of one of "kaiming" or "modified",
        # "hardcoded"
        assert dendrite_init in ("kaiming", "modified", "hardcoded")

        super().__init__()

        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw
        self.hardcode_dendrites = (dendrite_init == "hardcoded")

        # Forward layers & k-winners
        self.dend1 = dendritic_layer_class(
            module=nn.Linear(input_size, hidden_size, bias=True),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=0.95,
            dendrite_sparsity=0.0 if self.hardcode_dendrites else 0.95,
        )
        self.dend2 = dendritic_layer_class(
            module=nn.Linear(hidden_size, hidden_size, bias=True),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=0.95,
            dendrite_sparsity=0.0 if self.hardcode_dendrites else 0.95,
        )
        self.classifier = SparseWeights(module=nn.Linear(hidden_size, output_size),
                                        sparsity=0.95)

        if kw:

            print(f"Using k-Winners: 0.05 'on'")
            self.kw1 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)
            self.kw2 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)

        if weight_init == "modified":

            # Scale weights to be sampled from the new inititialization U(-h, h) where
            # h = sqrt(1 / (weight_density * previous_layer_percent_on))
            self._init_sparse_weights(self.dend1, 0.0)
            self._init_sparse_weights(self.dend2, 0.95 if kw else 0.0)
            self._init_sparse_weights(self.classifier, 0.95 if kw else 0.0)

        if dendrite_init == "modified":
            self._init_sparse_dendrites(self.dend1, 0.95)
            self._init_sparse_dendrites(self.dend2, 0.95)

        elif dendrite_init == "hardcoded":
            # Dendritic weights will not be updated during backward pass
            for name, param in self.named_parameters():
                if "segments" in name:
                    param.requires_grad = False

    def forward(self, x, context):
        output = self.dend1(x, context=context)
        output = self.kw1(output) if self.kw else output

        output = self.dend2(output, context=context)
        output = self.kw2(output) if self.kw else output

        output = self.classifier(output)
        return output

    # ------ Weight initialization functions
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
        fan_in = m.segments.dim_context
        bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
        nn.init.uniform_(m.segments.weights, -bound, bound)
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
        self._hardcode_dendritic_weights(self.dend1.segments, context_vectors, init)
        self._hardcode_dendritic_weights(self.dend2.segments, context_vectors, init)

    @staticmethod
    def _hardcode_dendritic_weights(dendrite_segments, context_vectors, init):
        num_units, num_segments, dim_context = dendrite_segments.weights.size()
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

        dendrite_segments.weights.data = new_dendritic_weights
