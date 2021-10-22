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
    AbsoluteMaxGatingDendriticLayer,
    AbsoluteMaxGatingDendriticLayer2d,
    BiasingDendriticLayer,
    DendriteSegments,
    GatingDendriticLayer,
    GatingDendriticLayer2d,
)
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params


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

        weights = dendrite_segments.weights.data
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
        dendrite_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=True,
        )
        dendrite_layer.rezero_weights()

        batch_size = 8
        input_dim = dendrite_layer.module.weight.shape[1]
        context_dim = dendrite_layer.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendrite_layer(x, context)
        self.assertEqual(out.shape, (8, 10))

    def test_forward_output_values(self):
        """
        Test all parts of the forward pass of a biasing dendritic layer from end to end.
        """

        # Dendritic weights as a bias.
        linear = torch.nn.Linear(4, 4, bias=False)
        dendrite_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=3,
            dim_context=4,
            module_sparsity=0.7,
            dendrite_sparsity=0.7,
            dendrite_bias=False,
        )
        dendrite_layer.rezero_weights()

        linear.weight.data[:] = torch.tensor(
            [
                [-0.04, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, -0.26],
                [0.00, 0.00, 0.00, -0.13],
                [0.00, 0.00, 0.00, 0.41],
            ],
            requires_grad=True,
        )

        dendrite_layer.segments.weights.data[:] = torch.tensor(
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

        # Expected dendrite activations: dendrite_layer.segments(context)
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
        actual_dendrite_activations = dendrite_layer.segments(context)
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
        actual_out = dendrite_layer(x, context)
        self.assertTrue(expected_out.allclose(actual_out))

    def test_apply_biasing_dendrites(self):
        """
        Validate the apply_dendrites function of a biasing dendritic layer.
        The max of the dendrite_activations should be taken per batch per unit.
        """
        # Dendritic weights as a bias.
        linear = torch.nn.Linear(10, 10)
        dendrite_layer = BiasingDendriticLayer(
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
        actual_output = dendrite_layer.apply_dendrites(y, dendrite_activations)

        all_matches = (expected_output == actual_output).all()
        self.assertTrue(all_matches)

    def test_sparsity(self):
        """
        Ensure both the linear weights and segment weights are rezeroed properly.
        """
        linear_sparsity = 70 / 100
        dendrite_sparsity = 13 / 15
        linear = torch.nn.Linear(10, 10)
        dendrite_layer = BiasingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=linear_sparsity,
            dendrite_sparsity=dendrite_sparsity,
            dendrite_bias=True,
        )

        linear_weights = linear.weight.data
        dendrite_weights = dendrite_layer.segments.weights.data
        linear_weights[:] = 1
        dendrite_weights[:] = 1
        dendrite_layer.rezero_weights()

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
        dendrite_layer = GatingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=True,
        )
        dendrite_layer.rezero_weights()

        batch_size = 8
        input_dim = dendrite_layer.module.weight.shape[1]
        context_dim = dendrite_layer.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendrite_layer(x, context)
        self.assertEqual(out.shape, (8, 10))

    def test_apply_gating_dendrites(self):
        """
        Validate the apply_dendrites function of a gating dendrite layer.
        The max of the dendrite_activations should be taken per batch per unit.
        """
        # Dendrite weights as a bias.
        linear = torch.nn.Linear(10, 10)
        dendrite_layer = GatingDendriticLayer(
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
        expected_output = y * torch.sigmoid(max_activation)
        actual_output = dendrite_layer.apply_dendrites(y, dendrite_activations)

        all_matches = (expected_output == actual_output).all()
        self.assertTrue(all_matches)


class AbsoluteMaxGatingDendriticLayerTests(unittest.TestCase):
    def test_forward_output_shape(self):
        """Validate shape of forward output."""
        linear = torch.nn.Linear(10, 10)
        dendrite_layer = AbsoluteMaxGatingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9
        )
        dendrite_layer.rezero_weights()

        batch_size = 8
        input_dim = dendrite_layer.module.weight.shape[1]
        context_dim = dendrite_layer.segments.weights.shape[2]
        x = torch.rand(batch_size, input_dim)
        context = torch.rand(batch_size, context_dim)

        out = dendrite_layer(x, context)
        self.assertEqual(out.shape, (8, 10))

    def test_segment_sparsity(self):
        """Test sparsity of dendritic segments."""
        linear = torch.nn.Linear(10, 11)
        dendrite_layer = AbsoluteMaxGatingDendriticLayer(
            module=linear,
            num_segments=10,
            dim_context=100,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        params, nonzero_params = count_nonzero_params(dendrite_layer.segments)
        self.assertAlmostEqual(0.1, nonzero_params / params)
        self.assertEqual(1100, nonzero_params)

    def test_apply_gating_dendrites(self):
        """
        Validate the outputs of the absolute max gating layer against hand-computed
        outputs.
        """
        linear = torch.nn.Linear(10, 10)
        dendrite_layer = AbsoluteMaxGatingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # pseudo output: batch_size=2, out_features=3
        y = torch.tensor([[0.1, -0.1, 0.5], [0.2, 0.3, -0.2]])

        # pseudo dendrite_activations: batch_size=2, num_units=3, num_segments=3
        dendrite_activations = torch.tensor(
            [
                [[0.43, -1.64, 1.49], [-0.79, 0.53, 1.08], [0.02, 0.04, -0.57]],
                [[1.79, -0.48, -0.38], [-0.15, 0.76, -1.13], [1.04, -0.58, -0.31]],
            ]
        )

        # Expected absolute max activation per batch per unit
        absolute_max_activations = torch.tensor([
            [-1.64, 1.08, -0.57],
            [1.79, -1.13, 1.04]
        ])

        # Expected output: dendrites applied as bias
        expected_output = y * torch.sigmoid(absolute_max_activations)
        actual_output = dendrite_layer.apply_dendrites(y, dendrite_activations)

        all_matches = (expected_output == actual_output).all()
        self.assertTrue(all_matches)

    def test_gradients(self):
        """
        Validate gradient values to ensure they are flowing through the absolute max
        operation. Note that this test doesn't actually consider the values of
        gradients, apart from whether they are zero or non-zero.
        """
        linear = torch.nn.Linear(10, 10)
        dendrite_layer = AbsoluteMaxGatingDendriticLayer(
            module=linear,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # pseudo output: batch_size=2, out_features=3
        y = torch.tensor([[0.1, -0.1, 0.5], [0.2, 0.3, -0.2]])

        # pseudo dendrite_activations: batch_size=2, num_units=3, num_segments=3
        dendrite_activations = torch.tensor(
            [
                [[0.43, -1.64, 1.49], [-0.79, 0.53, 1.08], [0.02, 0.04, -0.57]],
                [[1.79, -0.48, -0.38], [-0.15, 0.76, -1.13], [1.04, -0.58, -0.31]],
            ], requires_grad=True
        )

        output = dendrite_layer.apply_dendrites(y, dendrite_activations)
        output.sum().backward()

        # Expected gradient mask
        expected_grad_mask = torch.tensor(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            ]
        )
        actual_grad_mask = 1.0 * (dendrite_activations.grad != 0.0)

        all_matches = (expected_grad_mask == actual_grad_mask).all()
        self.assertTrue(all_matches)


class GatingDendriticLayer2dTests(unittest.TestCase):
    def test_forward(self):
        """ Validate the output values of the output tensor returned by `forward`. """

        # Initialize convolutional layer
        conv_layer = torch.nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=2, stride=1, bias=True
        )

        # Initialize dendrite layer
        dendrite_layer = GatingDendriticLayer2d(
            module=conv_layer,
            num_segments=3,
            dim_context=4,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # Set weights and biases of convolutional layer
        conv_layer.weight.data[:] = torch.tensor(
            [
                [
                    [[0.0000, 0.3105], [-0.1523, 0.0000]],
                    [[0.0000, 0.0083], [-0.2167, 0.0483]]
                ],
                [
                    [[0.1621, 0.0000], [-0.3283, 0.0101]],
                    [[-0.1045, 0.0261], [0.0000, 0.0000]]
                ],
                [
                    [[0.0000, -0.0968], [0.0499, 0.0000]],
                    [[0.0850, 0.0000], [0.2646, -0.3485]]
                ]
            ], requires_grad=True
        )
        conv_layer.bias.data[:] = torch.tensor(
            [-0.2027, -0.1821, 0.2152], requires_grad=True
        )

        # Dendrite weights: num_channels=3, num_segments=3, dim_context=4
        dendrite_layer.segments.weights.data[:] = torch.tensor([
            [
                [-0.4933, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3805, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.1641]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.3555],
                [0.0000, 0.0000, 0.0000, 0.1892],
                [0.0000, 0.0000, -0.4274, 0.0000]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.0957],
                [0.0000, 0.0000, -0.0689, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.3192]
            ]
        ])

        # Input to dendrite layer: batch_size=2, num_channels=2, width=3, height=3
        x = torch.tensor([
            [
                [
                    [0.1553, 0.3405, 0.2367],
                    [0.7661, 0.1383, 0.6675],
                    [0.6464, 0.1559, 0.9777]
                ],
                [
                    [0.4114, 0.6362, 0.7020],
                    [0.2617, 0.2275, 0.4238],
                    [0.6374, 0.8270, 0.7528]
                ]
            ],
            [
                [
                    [0.8331, 0.7792, 0.4369],
                    [0.7947, 0.2609, 0.1992],
                    [0.1527, 0.3006, 0.5496]
                ],
                [
                    [0.6811, 0.6871, 0.0148],
                    [0.6084, 0.8351, 0.5382],
                    [0.7421, 0.8639, 0.7444]
                ]
            ]
        ])

        # Context input to dendrite layer: batch_size=2, dim_context=4
        context_vectors = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # Expected max dendrite activations (pre-sigmoid):
        # [[0.3805  zero  zero]
        #  [  zero  zero  zero]]

        # Expected output of convolutional layer:
        #
        # batch item 1 (each row corresponds to an output channel)
        # [[[-0.2541 -0.1733] [-0.3545 -0.1585]]
        #  [[-0.4334 -0.2137] [-0.2900 -0.2137]]
        #  [[ 0.2454  0.1658] [ 0.1368  0.1342]]]
        #
        # batch item 2
        # [[[-0.1676 -0.2616] [-0.2571 -0.3334]]
        #  [[-0.3586 -0.2108] [-0.1422 -0.3062]]
        #  [[ 0.1073  0.2777] [ 0.1446  0.2511]]]

        # Overall expected output of dendrite layer:
        #
        # batch item 1 (each row corresponds to an output channel)
        # [[[-0.1509338 -0.10293911] [-0.21057076 -0.094148]]
        #  [[-0.2167    -0.10685   ] [-0.145      -0.10685 ]]
        #  [[ 0.1227     0.0829    ] [ 0.0684      0.0671  ]]]
        #
        # batch item 2
        # [[[-0.0838 -0.1308] [-0.1285 -0.1667]]
        #  [[-0.1793 -0.1054] [-0.0711 -0.1531]]
        #  [[ 0.0536  0.1389] [ 0.0723  0.1256]]]

        expected_output = torch.tensor([
            [
                [[-0.1509338, -0.10293911], [-0.21057076, -0.094148]],
                [[-0.2167, -0.10685], [-0.145, -0.10685]],
                [[0.1227, 0.0829], [0.0684, 0.0671]]
            ],
            [
                [[-0.0838, -0.1308], [-0.1285, -0.1667]],
                [[-0.1793, -0.1054], [-0.0711, -0.1531]],
                [[0.0536, 0.1389], [0.0723, 0.1256]]
            ]
        ])

        actual_output = dendrite_layer(x, context_vectors)
        self.assertTrue(torch.allclose(expected_output, actual_output, atol=1e-4))

    def test_apply_gating_dendrites(self):
        conv_layer = torch.nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, stride=1, bias=True
        )
        dendrite_layer = GatingDendriticLayer2d(
            module=conv_layer,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # pseudo output: batch_size=2, num_channels=3, height=2, width=2
        y = torch.tensor([
            [
                [[0.3, 0.4], [-0.2, 0.1]],
                [[-0.3, 0.5], [-0.1, 0.1]],
                [[0.0, 0.1], [0.3, 0.2]]
            ],
            [
                [[0.1, -0.2], [-0.2, 0.1]],
                [[0.0, 0.1], [-0.4, -0.1]],
                [[-0.3, 0.0], [0.2, 0.4]]
            ],
        ])

        # pseudo dendrite_activations: batch_size=2, num_channels=3, num_segments=3
        dendrite_activations = torch.tensor(
            [
                [[0.4, 0.9, -0.1], [-0.8, 0.7, 0.0], [0.6, -0.6, -0.7]],
                [[0.2, 0.8, 0.8], [-0.1, -0.4, -0.5], [0.0, 0.0, 0.0]],
            ]
        )

        # Expected max dendrite activations:
        # [[0.9   0.7   0.6]
        #  [0.8  -0.1   0.0]]

        # Expected output based on `dendrite_activations`
        expected_output = torch.tensor([
            [
                [[0.2133, 0.2844], [-0.1422, 0.0711]],
                [[-0.2005, 0.3341], [-0.0668, 0.0668]],
                [[0.0, 0.0646], [0.1937, 0.1291]]
            ],
            [
                [[0.0690, -0.1380], [-0.1380, 0.0690]],
                [[0.0, 0.0475], [-0.1900, -0.0475]],
                [[-0.15, 0.0], [0.1, 0.2]]
            ],
        ])

        actual_output = dendrite_layer.apply_dendrites(y, dendrite_activations)
        all_matches = torch.allclose(expected_output, actual_output, atol=1e-4)
        self.assertTrue(all_matches)

    def test_gradients(self):
        """
        Ensure dendrite gradients are flowing through the layer
        `GatingDendriticLayer2d`. Note that this test doesn't actually consider the
        values of gradients, apart from whether they are zero or non-zero.
        """
        conv_layer = torch.nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=2, stride=1, bias=True
        )
        dendrite_layer = GatingDendriticLayer2d(
            module=conv_layer,
            num_segments=3,
            dim_context=4,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # Dendrite weights: num_channels=3, num_segments=3, dim_context=4
        dendrite_layer.segments.weights.data[:] = torch.tensor([
            [
                [-0.4933, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3805, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.1641]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.3555],
                [0.0000, 0.0000, 0.0000, 0.1892],
                [0.0000, 0.0000, -0.4274, 0.0000]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.0957],
                [0.0000, 0.0000, -0.0689, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.3192]
            ]
        ])

        # Input to dendrite layer: batch_size=1, num_channels=2, width=3, height=3
        x = torch.randn((1, 2, 3, 3))

        # Context input to dendrite layer: batch_size=1, dim_context=4
        context_vectors = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        # Expected dendrite activations:
        #
        # batch item 1 (each row corresponds to an output channel)
        # [[-0.4933  0.3805    zero]
        #  [   zero    zero -0.4274]
        #  [   zero -0.0689    zero]]

        # Expected dendrite gradient mask
        #
        # batch item 1
        # [[0  1  0]
        #  [1  0  0]
        #  [1  0  0]]

        output = dendrite_layer(x, context_vectors)
        output.sum().backward()

        # Expected gradient mask
        expected_grad_mask = torch.tensor([
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ])
        actual_grad_mask = 1.0 * (dendrite_layer.segments.weights.grad != 0.0)

        all_matches = (expected_grad_mask == actual_grad_mask).all()
        self.assertTrue(all_matches)


class AbsoluteMaxGatingDendriticLayer2dTests(unittest.TestCase):
    def test_forward(self):
        """ Validate the output values of the output tensor returned by `forward`. """

        # Initialize convolutional layer
        conv_layer = torch.nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=2, stride=1, bias=True
        )

        # Initialize dendrite layer
        dendrite_layer = AbsoluteMaxGatingDendriticLayer2d(
            module=conv_layer,
            num_segments=3,
            dim_context=4,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # Set weights and biases of convolutional layer
        conv_layer.weight.data[:] = torch.tensor(
            [
                [
                    [[0.0000, 0.3105], [-0.1523, 0.0000]],
                    [[0.0000, 0.0083], [-0.2167, 0.0483]]
                ],
                [
                    [[0.1621, 0.0000], [-0.3283, 0.0101]],
                    [[-0.1045, 0.0261], [0.0000, 0.0000]]
                ],
                [
                    [[0.0000, -0.0968], [0.0499, 0.0000]],
                    [[0.0850, 0.0000], [0.2646, -0.3485]]
                ]
            ], requires_grad=True
        )
        conv_layer.bias.data[:] = torch.tensor(
            [-0.2027, -0.1821, 0.2152], requires_grad=True
        )

        # Dendrite weights: num_channels=3, num_segments=3, dim_context=4
        dendrite_layer.segments.weights.data[:] = torch.tensor([
            [
                [-0.4933, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3805, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.1641]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.3555],
                [0.0000, 0.0000, 0.0000, 0.1892],
                [0.0000, 0.0000, -0.4274, 0.0000]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.0957],
                [0.0000, 0.0000, -0.0689, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.3192]
            ]
        ])

        # Input to dendrite layer: batch_size=2, num_channels=2, width=3, height=3
        x = torch.tensor([
            [
                [
                    [0.1553, 0.3405, 0.2367],
                    [0.7661, 0.1383, 0.6675],
                    [0.6464, 0.1559, 0.9777]
                ],
                [
                    [0.4114, 0.6362, 0.7020],
                    [0.2617, 0.2275, 0.4238],
                    [0.6374, 0.8270, 0.7528]
                ]
            ],
            [
                [
                    [0.8331, 0.7792, 0.4369],
                    [0.7947, 0.2609, 0.1992],
                    [0.1527, 0.3006, 0.5496]
                ],
                [
                    [0.6811, 0.6871, 0.0148],
                    [0.6084, 0.8351, 0.5382],
                    [0.7421, 0.8639, 0.7444]
                ]
            ]
        ])

        # Context input to dendrite layer: batch_size=2, dim_context=4
        context_vectors = torch.tensor([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # Expected absolute max dendrite activations (pre-sigmoid):
        # [[-0.4933 -0.4274 -0.0689]
        #  [   zero    zero    zero]]

        # Expected output of convolutional layer:
        #
        # batch item 1 (each row corresponds to an output channel)
        # [[[-0.2541 -0.1733] [-0.3545 -0.1585]]
        #  [[-0.4334 -0.2137] [-0.2900 -0.2137]]
        #  [[ 0.2454  0.1658] [ 0.1368  0.1342]]]
        #
        # batch item 2
        # [[[-0.1676 -0.2616] [-0.2571 -0.3334]]
        #  [[-0.3586 -0.2108] [-0.1422 -0.3062]]
        #  [[ 0.1073  0.2777] [ 0.1446  0.2511]]]

        # Overall expected output of dendrite layer:
        #
        # batch item 1 (each row corresponds to an output channel)
        # [[[-0.0963335  -0.06570089] [-0.13439679 -0.06008996]]
        #  [[-0.17108351 -0.08435751] [-0.11447673 -0.08435751]]
        #  [[ 0.11847466  0.08004522] [ 0.06604455  0.06478932]]]
        #
        # batch item 2
        # [[[-0.0838 -0.1308] [-0.1285 -0.1667]]
        #  [[-0.1793 -0.1054] [-0.0711 -0.1531]]
        #  [[ 0.0536  0.1389] [ 0.0723  0.1256]]]

        expected_output = torch.tensor([
            [
                [[-0.0963335, -0.06570089], [-0.13439679, -0.06008996]],
                [[-0.17108351, -0.08435751], [-0.11447673, -0.08435751]],
                [[0.11847466, 0.08004522], [0.06604455, 0.06478932]]
            ],
            [
                [[-0.0838, -0.1308], [-0.1285, -0.1667]],
                [[-0.1793, -0.1054], [-0.0711, -0.1531]],
                [[0.0536, 0.1389], [0.0723, 0.1256]]
            ]
        ])

        actual_output = dendrite_layer(x, context_vectors)
        self.assertTrue(torch.allclose(expected_output, actual_output, atol=1e-4))

    def test_apply_gating_dendrites(self):
        conv_layer = torch.nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, stride=1, bias=True
        )
        dendrite_layer = AbsoluteMaxGatingDendriticLayer2d(
            module=conv_layer,
            num_segments=20,
            dim_context=15,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # pseudo output: batch_size=2, num_channels=3, height=2, width=2
        y = torch.tensor([
            [
                [[0.3, 0.4], [-0.2, 0.1]],
                [[-0.3, 0.5], [-0.1, 0.1]],
                [[0.0, 0.1], [0.3, 0.2]]
            ],
            [
                [[0.1, -0.2], [-0.2, 0.1]],
                [[0.0, 0.1], [-0.4, -0.1]],
                [[-0.3, 0.0], [0.2, 0.4]]
            ],
        ])

        # pseudo dendrite_activations: batch_size=2, num_channels=3, num_segments=3
        dendrite_activations = torch.tensor(
            [
                [[0.4, 0.9, -0.1], [-0.8, 0.7, 0.0], [0.6, -0.6, -0.7]],
                [[0.2, 0.8, 0.8], [-0.1, -0.4, 0.5], [0.0, 0.0, 0.0]],
            ]
        )

        # Expected absolute max dendrite activations (pre-sigmoid):
        # [[0.9  -0.8  -0.7]
        #  [0.8   0.5   0.0]]

        # Expected output based on `dendrite_activations`
        expected_output = torch.tensor([
            [
                [[0.2133, 0.2844], [-0.1422, 0.0711]],
                [[-0.093, 0.155], [-0.031, 0.031]],
                [[0.0, 0.0332], [0.0995, 0.0664]]
            ],
            [
                [[0.069, -0.138], [-0.138, 0.069]],
                [[0.0, 0.0622], [-0.249, -0.0622]],
                [[-0.15, 0.0], [0.1, 0.2]]
            ],
        ])

        actual_output = dendrite_layer.apply_dendrites(y, dendrite_activations)
        all_matches = torch.allclose(expected_output, actual_output, atol=1e-4)
        self.assertTrue(all_matches)

    def test_gradients(self):
        """
        Ensure dendrite gradients are flowing through the layer
        `AbsoluteMaxGatingDendriticLayer2d`. Note that this test doesn't actually
        consider the values of gradients, apart from whether they are zero or non-zero.
        """
        conv_layer = torch.nn.Conv2d(
            in_channels=2, out_channels=3, kernel_size=2, stride=1, bias=True
        )
        dendrite_layer = AbsoluteMaxGatingDendriticLayer2d(
            module=conv_layer,
            num_segments=3,
            dim_context=4,
            module_sparsity=0.7,
            dendrite_sparsity=0.9,
            dendrite_bias=False,
        )

        # Dendrite weights: num_channels=3, num_segments=3, dim_context=4
        dendrite_layer.segments.weights.data[:] = torch.tensor([
            [
                [-0.4933, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3805, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.1641]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.3555],
                [0.0000, 0.0000, 0.0000, 0.1892],
                [0.0000, 0.0000, -0.4274, 0.0000]
            ],
            [
                [0.0000, 0.0000, 0.0000, 0.0957],
                [0.0000, 0.0000, -0.0689, 0.0000],
                [0.0000, 0.0000, 0.0000, -0.3192]
            ]
        ])

        # Input to dendrite layer: batch_size=1, num_channels=2, width=3, height=3
        x = torch.randn((1, 2, 3, 3))

        # Context input to dendrite layer: batch_size=1, dim_context=4
        context_vectors = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        # Expected dendrite activations:
        #
        # batch item 1 (each row corresponds to an output channel)
        # [[-0.4933  0.3805    zero]
        #  [   zero    zero -0.4274]
        #  [   zero -0.0689    zero]]

        # Expected dendrite gradient mask
        #
        # batch item 1
        # [[1  0  0]
        #  [0  0  1]
        #  [0  1  0]]

        output = dendrite_layer(x, context_vectors)
        output.sum().backward()

        # Expected gradient mask
        expected_grad_mask = torch.tensor([
            [[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        ])
        actual_grad_mask = 1.0 * (dendrite_layer.segments.weights.grad != 0.0)

        all_matches = (expected_grad_mask == actual_grad_mask).all()
        self.assertTrue(all_matches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
