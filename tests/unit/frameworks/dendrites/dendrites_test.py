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
from torch.nn.functional import sigmoid

from nupic.research.frameworks.dendrites import (
    BiasingDendriticLayer,
    DendriteSegments,
    GatingDendriticLayer,
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
                self.assertEqual(
                    sparsity,
                    actual_sparsity,
                    f"Sparsity {actual_sparsity} != {sparsity}"
                    f"for unit {unit} and segment {segment}",
                )

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
            self.assertTrue(
                same_out,
                f"Didn't observe the expected output for unit {unit}: "
                f"actual_out - expected_out = {actual_out - expected_out}",
            )


class BiasingDendriticLayerTests(unittest.TestCase):
    def test_forward_output_shape(self):
        """Validate shape of forward output."""

        # Dendritic weights as a bias.
        linear = torch.nn.Linear(10, 10)
        dendritic_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=True,
        )
        dendritic_layer.rezero_weights()

        batch_size = 8
        input_dim = dendritic_layer.module.weight.shape[1]
        context_dim = dendritic_layer.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendritic_layer(x, context)
        self.assertEqual(out.shape, (8, 10))

    def test_forward_output_values(self):
        """
        Test all parts of the forward pass of a biasing dendritic layer from end to end.
        """

        # Dendritic weights as a bias.
        linear = torch.nn.Linear(4, 4, bias=False)
        dendritic_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=3,
            dim_context=4,
            module_sparsity=0.7,
            dendrite_sparsity=0.7,
            dendrite_bias=False,
        )
        dendritic_layer.rezero_weights()

        linear.weight.data[:] = torch.tensor(
            [
                [-0.04, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, -0.26],
                [0.00, 0.00, 0.00, -0.13],
                [0.00, 0.00, 0.00, 0.41],
            ],
            requires_grad=True,
        )

        dendritic_layer.segments.weights.data[:] = torch.tensor(
            [
                [
                    [-0.26, 0.00, 0.00, 0.00],
                    [0.09, 0.00, 0.00, 0.00],
                    [-0.34, 0.00, 0.00, 0.00],
                ],
                [
                    [0.00, 0.00, 0.00, 0.36],
                    [0.00, 0.00, 0.00, -0.32],
                    [0.00, 0.00, 0.00, 0.41],
                ],
                [
                    [0.00, 0.00, 0.00, 0.18],
                    [0.00, 0.00, 0.38, 0.00],
                    [0.00, 0.00, 0.23, 0.00],
                ],
                [
                    [0.00, 0.00, 0.00, 0.23],
                    [-0.30, 0.00, 0.00, 0.00],
                    [0.00, 0.00, -0.24, 0.00],
                ],
            ],
            requires_grad=True,
        )

        # Pseudo input: batch_size=2, input_dim=4
        x = torch.tensor([[0.79, 0.36, 0.47, 0.30], [0.55, 0.64, 0.50, 0.50]])

        # Pseudo input: batch_size=2, context_dim=4
        context = torch.tensor([[0.84, 0.63, 0.67, 0.42], [0.30, 0.07, 0.52, 0.15]])

        # Expected dendrite activations: dendritic_layer.segments(context)
        # This will be the shape batch_size x num_units x num_segments
        expected_dendrite_activations = torch.tensor(
            [
                [
                    [-0.2184, 0.0756, -0.2856],
                    [0.1512, -0.1344, 0.1722],
                    [0.0756, 0.2546, 0.1541],
                    [0.0966, -0.2520, -0.1608],
                ],
                [
                    [-0.0780, 0.0270, -0.1020],
                    [0.0540, -0.0480, 0.0615],
                    [0.0270, 0.1976, 0.1196],
                    [0.0345, -0.0900, -0.1248],
                ],
            ]
        )

        # Validate dendrite activations.
        actual_dendrite_activations = dendritic_layer.segments(context)
        self.assertTrue(
            expected_dendrite_activations.allclose(actual_dendrite_activations)
        )

        # Validate the biasing term: max per batch per unit
        biasing_dendrites = torch.tensor(
            [[0.0756, 0.1722, 0.2546, 0.0966], [0.0270, 0.0615, 0.1976, 0.0345]]
        )
        all_matches = (
            expected_dendrite_activations.max(dim=2).values == biasing_dendrites
        ).all()
        self.assertTrue(all_matches)

        # Validate output of dendritic layer.
        expected_out = linear(x) + biasing_dendrites
        actual_out = dendritic_layer(x, context)
        self.assertTrue(expected_out.allclose(actual_out))

    def test_apply_biasing_dendrites(self):
        """
        Validate the apply_dendrites function of a biasing dendritic layer.
        The max of the dendrite_activations should be taken per batch per unit.
        """
        # Dendritic weights as a bias.
        linear = torch.nn.Linear(10, 10)
        dendritic_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=True,
        )

        # pseudo output: batch_size=2, out_features=3
        y = torch.tensor([[0.1, -0.1, 0.5], [0.2, 0.3, -0.2]])

        # pseudo dendrite_activations: batch_size=2, num_units=3, num_segments=3
        dendrite_activations = torch.tensor(
            [
                [[0.43, 1.64, 1.49], [-0.79, 0.53, 1.08], [0.02, 0.04, 0.57]],
                [[1.79, -0.48, -0.38], [-0.15, 0.76, -1.13], [1.04, -0.58, -0.31]],
            ]
        )

        # Expected max activation per batch per unit.
        max_activation = torch.tensor([[1.64, 1.08, 0.57], [1.79, 0.76, 1.04]])

        # Expected output: dendrites applied as bias
        expected_output = y + max_activation
        actual_output = dendritic_layer.apply_dendrites(y, dendrite_activations)

        all_matches = (expected_output == actual_output).all()
        self.assertTrue(all_matches)

    def test_sparsity(self):
        """
        Ensure both the linear weights and segment weights are rezeroed properly.
        """
        linear_sparsity = 70 / 100
        dendrite_sparsity = 13 / 15
        linear = torch.nn.Linear(10, 10)
        dendritic_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=linear_sparsity,
            dendrite_sparsity=dendrite_sparsity,
            dendrite_bias=True,
        )

        linear_weights = linear.weight.data
        dendrite_weights = dendritic_layer.segments.weights
        linear_weights[:] = 1
        dendrite_weights[:] = 1
        dendritic_layer.rezero_weights()

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
        dendritic_layer = GatingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=True,
        )
        dendritic_layer.rezero_weights()

        batch_size = 8
        input_dim = dendritic_layer.module.weight.shape[1]
        context_dim = dendritic_layer.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendritic_layer(x, context)
        self.assertEqual(out.shape, (8, 10))

    def test_apply_gating_dendrites(self):
        """
        Validate the apply_dendrites function of a gating dendritic layer.
        The max of the dendrite_activations should be taken per batch per unit.
        """
        # Dendritic weights as a bias.
        linear = torch.nn.Linear(10, 10)
        dendritic_layer = GatingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=True,
        )

        # pseudo output: batch_size=2, out_features=3
        y = torch.tensor([[0.73, 0.72, 0.62], [0.26, 0.24, 0.65]])

        # pseudo dendrite_activations: batch_size=2, num_units=3, num_segments=2
        dendrite_activations = torch.tensor(
            [
                [[-1.15, -0.49], [0.87, -0.58], [-0.36, -0.93]],
                [[-0.08, -1.00], [-0.71, 0.08], [0.15, 0.40]],
            ]
        )

        # Expected max activation per batch per unit.
        max_activation = torch.tensor([[-0.49, 0.87, -0.36], [-0.08, 0.08, 0.40]])

        # Expected output: dendrites applied as gate
        expected_output = y * sigmoid(max_activation)
        actual_output = dendritic_layer.apply_dendrites(y, dendrite_activations)

        all_matches = (expected_output == actual_output).all()
        self.assertTrue(all_matches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
