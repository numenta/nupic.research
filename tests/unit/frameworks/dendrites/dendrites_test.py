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

import unittest

import torch

from nupic.research.frameworks.dendrites import (
    DendriteSegments,
    DendriticWeights,
    GatingDendriticWeights,
)


class DendriteSegmentsTests(unittest.TestCase):

    def test_forward(self):
        dendrite_segment = DendriteSegments(
            num_units=10, num_segments=20, dim_context=15, sparsity=0.7, bias=True
        )
        dendrite_segment.rezero_weights()

        batch_size = 8
        context = torch.rand(batch_size, dendrite_segment.dim_context)
        out = dendrite_segment(context)
        self.assertEqual(out.shape, (8, 10, 20))


class DendriticWeightsTests(unittest.TestCase):

    def test_forward(self):
        # Dendritic weights as a bias.
        linear = torch.nn.Linear(10, 10)
        dendritic_weights = DendriticWeights(
            module=linear, num_segments=20, dim_context=15,
            module_sparsity=0.7, dendrite_sparsity=0.9, dendrite_bias=True
        )
        dendritic_weights.rezero_weights()

        batch_size = 8
        input_dim = dendritic_weights.module.weight.shape[1]
        context_dim = dendritic_weights.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendritic_weights(x, context)
        self.assertEqual(out.shape, (8, 10))

    def test_gating_forward(self):
        # Gating dendritic weights.
        linear = torch.nn.Linear(10, 10)
        dendritic_weights = DendriticWeights(
            module=linear, num_segments=20, dim_context=15,
            module_sparsity=0.7, dendrite_sparsity=0.9, dendrite_bias=True
        )
        dendritic_weights.rezero_weights()

        batch_size = 8
        input_dim = dendritic_weights.module.weight.shape[1]
        context_dim = dendritic_weights.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendritic_weights(x, context)
        self.assertEqual(out.shape, (8, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
