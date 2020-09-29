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
        """Validate shape of forward output."""

        dendrite_segments = DendriteSegments(
            num_units=10, num_segments=20, dim_context=15, sparsity=0.7, bias=True
        )
        dendrite_segments.rezero_weights()

        batch_size = 8
        context = torch.rand(batch_size, dendrite_segments.dim_context)
        out = dendrite_segments(context)
        self.assertEqual(out.shape, (8, 10, 20))

    def test_sparsity(self):
        """
        Validate that sparsity is enforced per unit per segement.
        """
        sparsity = 9 / 15
        dendrite_segments = DendriteSegments(
            num_units=10, num_segments=20, dim_context=15, sparsity=sparsity, bias=True
        )

        weights = dendrite_segments.weights
        weights[:] = 1
        dendrite_segments.rezero_weights()

        for unit in range(dendrite_segments.num_units):
            for segment in range(dendrite_segments.num_segments):
                w = weights[unit, segment, :]
                num_off = (weights[unit, segment, :] == 0).sum().item()
                actual_sparsity = num_off / w.numel()
                self.assertEqual(sparsity, actual_sparsity,
                                 f"Sparsity {actual_sparsity} != {sparsity}"
                                 f"for unit {unit} and segment {segment}")

    def test_equivalent_forward(self):
        """
        Validate output with respect to an equivalent operation:
        applying the dendrite segments one-by-one for each unit.
        """
        dendrite_segments = DendriteSegments(
            num_units=10, num_segments=20, dim_context=15, sparsity=0.7, bias=True
        )

        batch_size = 8
        context = torch.rand(batch_size, dendrite_segments.dim_context)
        out = dendrite_segments(context)  # shape batch_size x num_units x num_segments

        weights = dendrite_segments.weights
        biases = dendrite_segments.biases
        for unit in range(dendrite_segments.num_units):
            unit_weight = weights[unit, ...]
            unit_bias = biases[unit, ...]

            expected_out = torch.nn.functional.linear(context, unit_weight, unit_bias)
            actual_out = out[:, unit, :]
            same_out = torch.allclose(actual_out, expected_out, atol=1e-7)
            self.assertTrue(same_out,
                            f"Didn't observe the expected output for unit {unit}: "
                            f"actual_out - expected_out = {actual_out - expected_out}")


class DendriticWeightsTests(unittest.TestCase):

    def test_forward(self):
        """Validate shape of forward output."""

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

    def test_sparsity(self):
        """
        Ensure both the linear weights and segment weights are rezeroed properly.
        """
        linear_sparsity = 70 / 100
        dendrite_sparsity = 13 / 15
        linear = torch.nn.Linear(10, 10)
        dendritic_weights = DendriticWeights(
            module=linear, num_segments=20, dim_context=15,
            module_sparsity=linear_sparsity, dendrite_sparsity=dendrite_sparsity,
            dendrite_bias=True
        )

        linear_weights = linear.weight.data
        dendrite_weights = dendritic_weights.segments.weights
        linear_weights[:] = 1
        dendrite_weights[:] = 1
        dendritic_weights.rezero_weights()

        num_off = (dendrite_weights == 0).sum().item()
        actual_dendrite_sparsity = num_off / dendrite_weights.numel()
        self.assertEqual(dendrite_sparsity, actual_dendrite_sparsity)

        num_off = (linear_weights == 0).sum().item()
        actual_linear_sparsity = num_off / linear_weights.numel()
        self.assertEqual(linear_sparsity, actual_linear_sparsity)

    def test_gating_forward(self):
        """Validate shape of forward output."""

        # Gating dendritic weights.
        linear = torch.nn.Linear(10, 10)
        dendritic_weights = GatingDendriticWeights(
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
