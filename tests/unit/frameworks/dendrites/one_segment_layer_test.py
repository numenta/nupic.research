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

from nupic.research.frameworks.dendrites import OneSegmentDendriticLayer
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params


class OneSegmentDendriticLayerTests(unittest.TestCase):
    def test_forward_output_shape(self):
        """Validate shape of forward output."""
        input_dim = 10
        context_dim = 15
        num_units = 11
        linear = torch.nn.Linear(input_dim, num_units)
        dendrite_layer = OneSegmentDendriticLayer(
            module=linear,
            num_segments=1,
            dim_context=context_dim,
            module_sparsity=0.7,
            dendrite_sparsity=0.9
        )

        batch_size = 8
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendrite_layer(x, context)
        self.assertEqual(out.shape, (batch_size, num_units))

    def test_segment_sparsity(self):
        """Validate shape of forward output."""
        linear = torch.nn.Linear(10, 11)
        dendrite_layer = OneSegmentDendriticLayer(
            module=linear,
            num_segments=1,
            dim_context=100,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        params, nonzero_params = count_nonzero_params(dendrite_layer.segments)
        self.assertAlmostEqual(0.1, nonzero_params / params)
        self.assertEqual(110, nonzero_params)

    def test_apply_gating_dendrites(self):
        """
        Validate the outputs of the absolute max gating layer against hand-computed
        outputs.
        """
        linear = torch.nn.Linear(10, 3)
        dendrite_layer = OneSegmentDendriticLayer(
            module=linear,
            num_segments=1,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # pseudo output: batch_size=2, out_features=3
        y = torch.tensor([[0.1, -0.1, 0.5], [0.2, 0.3, -0.2]])

        # pseudo dendrite_activations: batch_size=2, num_units=3
        dendrite_activations = torch.tensor(
            [
                [0.43, -1.64, 1.49],
                [1.79, -0.48, -0.38],
            ]
        )

        # Expected output: dendrites applied as bias
        expected_output = y * torch.sigmoid(dendrite_activations)

        actual_output = dendrite_layer.apply_dendrites(y, dendrite_activations)
        all_matches = (expected_output == actual_output).all()
        self.assertTrue(all_matches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
