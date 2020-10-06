# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

"""
A simple implementation of dendrite weights. This combines the output from a (sparse)
linear layer with the output from a set of dendritic segments.
"""

from torch.nn.functional import sigmoid

from nupic.research.frameworks.dendrites import DendriteSegments
from nupic.torch.modules.sparse_weights import SparseWeights


class BiasingDendriticLayer(SparseWeights):
    """
    This combines a DendriteSegments module with a SparseLinear module.
    The output from the dendrite segments (shape of num_units x num_segments)
    is applied to the output of of the linear weights (shape of num_units).
    Thus, each linear output unit gets modulated by a set of dendritic segments.
    """

    def __init__(
        self, module, num_segments, dim_context,
        module_sparsity, dendrite_sparsity, dendrite_bias=None
    ):
        """
        :param module: linear module from in-units to out-units
        :param num_segments: number of dendrite segments per out-unit
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param module_sparsity: sparsity applied over linear module;
        :param dendrite_sparsity: sparsity applied transformation per unit per segment
        """
        self.segments = None
        super().__init__(module, sparsity=module_sparsity)

        self.segments = DendriteSegments(
            num_units=module.weight.shape[0],
            num_segments=num_segments,
            dim_context=dim_context,
            sparsity=dendrite_sparsity,
            bias=dendrite_bias,
        )

        self.rezero_weights()

    def rezero_weights(self):
        super().rezero_weights()
        if self.segments is not None:  # only none at beggining of init
            self.segments.rezero_weights()

    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a bias."""
        return y + dendrite_activations.max(dim=2).values  # max along each segment

    def forward(self, x, context):
        """
        Compute of linear layer and apply output of dendrite segments.
        """
        y = super().forward(x)
        dendrite_activations = self.segments(context)  # num_units x num_segments
        return self.apply_dendrites(y, dendrite_activations)


class GatingDendriticLayer(BiasingDendriticLayer):
    def apply_dendrites(self, y, dendrite_activations):
        """Apply dendrites as a gating mechanism."""
        # Multiple by the sigmoid of the max along each segment.
        return y * sigmoid(dendrite_activations.max(dim=2).values)
