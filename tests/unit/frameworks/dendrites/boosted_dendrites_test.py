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

import unittest

import torch

from nupic.research.frameworks.dendrites import (
    BoostedDendritesAbsMaxGate1d,
    DendriteSegments,
)
from nupic.research.frameworks.pytorch.mask_utils import indices_to_mask

BATCH_SIZE = 2
NUM_UNITS = 3
NUM_SEGMENTS = 4
DIM_CONTEXT = 3


class BoostedDendritesTest(unittest.TestCase):
    """
    A test class for the `BoostedDendrites` modules.
    """

    def setUp(self):

        # An apply-dendrites module that boost dendritic activations and applies them to
        # modulate a given output in it's forward pass.
        self.apply_dendrites = BoostedDendritesAbsMaxGate1d(
            NUM_UNITS,
            NUM_SEGMENTS,
            boost_strength=1.0,
            boost_strength_factor=0.9,
            duty_cycle_period=4,
        )

        # A dendrite segments module that outputs dendritic activations.
        self.dendrite_segments = DendriteSegments(
            NUM_UNITS, NUM_SEGMENTS, DIM_CONTEXT, sparsity=0.5, bias=None
        )

        # Control the initial set of weights for reproducibility.
        self.dendrite_segments.weights.data[:] = torch.tensor(
            [
                [
                    [0.0000, 0.5414, -0.2743],
                    [-0.3823, -0.2690, 0.0000],
                    [0.0000, -0.3561, -0.5487],
                    [-0.5041, 0.0000, 0.2422],
                ],
                [
                    [-0.3117, 0.1230, 0.0000],
                    [-0.0781, 0.0000, 0.4663],
                    [0.3242, 0.5495, 0.0000],
                    [0.0000, -0.1231, 0.0379],
                ],
                [
                    [0.0000, -0.1057, 0.0344],
                    [0.4062, 0.0000, -0.5166],
                    [0.0000, -0.3438, -0.4910],
                    [0.1749, 0.0000, -0.5589],
                ],
            ],
            requires_grad=True,
        )

    def test_initialized_duty_cycles(self):
        duty_cycles = self.apply_dendrites.duty_cycles
        zeros = torch.zeros(NUM_UNITS, NUM_SEGMENTS)
        self.assertTrue((duty_cycles == zeros).all())

    def test_boosted_activations(self):
        """
        Test the expected values of boosted dendrites activations.
        """

        # Pseudo output from a Dendrite Segments module.
        dendrite_activations = torch.tensor(
            [
                [
                    [0.3539, -0.5627, -0.4905, -0.3609],
                    [-0.1737, 0.0956, 0.7448, -0.0891],
                    [-0.0759, 0.1772, -0.4598, -0.0426],
                ],
                [
                    [0.2239, -0.3209, -0.3698, -0.1477],
                    [-0.0672, 0.1080, 0.4547, -0.0584],
                    [-0.0495, 0.0209, -0.3452, -0.0937],
                ],
            ]
        )

        # Validate boost activations.
        expected_boosted = torch.tensor(
            [
                [
                    [0.3539, -0.5627, -0.4905, -0.3609],
                    [-0.1737, 0.0956, 0.7448, -0.0891],
                    [-0.0759, 0.1772, -0.4598, -0.0426],
                ],
                [
                    [0.2239, -0.3209, -0.3698, -0.1477],
                    [-0.0672, 0.1080, 0.4547, -0.0584],
                    [-0.0495, 0.0209, -0.3452, -0.0937],
                ],
            ]
        )

        boosted_activations = self.apply_dendrites.boost_activations(
            dendrite_activations
        )
        all_equal = (expected_boosted == boosted_activations).all()
        self.assertTrue(all_equal)

    def test_winning_indices(self):
        """
        Test the expected winning segments indices of a boosted dendrite activations.
        """

        # Pseudo output from a Dendrite Segments module.
        dendrite_activations = torch.tensor(
            [
                [
                    [0.3539, -0.5627, -0.4905, -0.3609],
                    [-0.1737, 0.0956, 0.7448, -0.0891],
                    [-0.0759, 0.1772, -0.4598, -0.0426],
                ],
                [
                    [0.2239, -0.3209, -0.3698, -0.1477],
                    [-0.0672, 0.1080, 0.4547, -0.0584],
                    [-0.0495, 0.0209, -0.3452, -0.0937],
                ],
            ]
        )

        # Pseudo output to be contextually modulated via dendrites.
        y = torch.tensor(
            [[0.5737, 0.0753, 0.0143], [0.5244, 0.7850, 0.5987]]
        )

        # Modulate y via dendrites with boosting enabled.
        y_modulated, winning_indices = self.apply_dendrites(y, dendrite_activations)
        winning_mask = indices_to_mask(
            winning_indices, (BATCH_SIZE, NUM_UNITS, NUM_SEGMENTS), dim=2
        )

        # Validate winning segments.
        expected_mask = torch.tensor(
            [
                [
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, True, False],
                ],
                [
                    [False, False, True, False],
                    [False, False, True, False],
                    [False, False, True, False],
                ],
            ]
        )

        all_equal = (expected_mask == winning_mask).all()
        self.assertTrue(all_equal)

    def test_duty_cycles_after_three_forward_passes(self):
        """
        Validate the duty cycles after three forward passes when the batch size is two
        and the duty cycle period is four.
        """

        # -----------
        # 1st Forward
        # -----------

        # Pseudo output from a Dendrite Segments module.
        dendrite_activations = torch.tensor(
            [
                [
                    [0.3539, -0.5627, -0.4905, -0.3609],
                    [-0.1737, 0.0956, 0.7448, -0.0891],
                    [-0.0759, 0.1772, -0.4598, -0.0426],
                ],
                [
                    [0.2239, -0.3209, -0.3698, -0.1477],
                    [-0.0672, 0.1080, 0.4547, -0.0584],
                    [-0.0495, 0.0209, -0.3452, -0.0937],
                ],
            ]
        )

        # Pseudo output to be contextually modulated via dendrites.
        y = torch.tensor(
            [[0.5737, 0.0753, 0.0143], [0.5244, 0.7850, 0.5987]]
        )

        # Modulate y via dendrites with boosting enabled.
        y_modulated, winning_indices = self.apply_dendrites(y, dendrite_activations)

        # Validate duty cycles.
        expected_duty_cycles = torch.tensor(
            [
                [0.0000, 0.5000, 0.5000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000, 0.0000],
            ]
        )
        actual_duty_cycles = self.apply_dendrites.duty_cycles
        all_equal = (actual_duty_cycles == expected_duty_cycles).all()
        self.assertTrue(all_equal)

        # -----------
        # 2nd Forward
        # -----------

        # Pseudo output from a Dendrite Segments module.
        # Pseudo output to be contextually modulated via dendrites.
        y = torch.tensor(
            [[0.5871, 0.9978, 0.5153], [0.8176, 0.8115, 0.5413]], requires_grad=True
        )
        dendrite_activations = torch.tensor(
            [
                [
                    [0.6483, 0.0565, 0.1324, 0.4405],
                    [0.6355, 0.1367, 0.3567, 0.2174],
                    [0.2007, 0.8239, 0.1598, 0.0793],
                ],
                [
                    [0.2204, 0.6325, 0.1207, 0.5185],
                    [0.2294, 0.9949, 0.7910, 0.8001],
                    [0.8268, 0.2881, 0.2339, 0.4337],
                ],
            ]
        )

        # Modulate y via dendrites with boosting enabled.
        y_modulated, winning_indices = self.apply_dendrites(y, dendrite_activations)

        # Validate duty cycles.
        expected_duty_cycles = torch.tensor(
            [
                [0.2500, 0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.5000, 0.0000],
                [0.2500, 0.2500, 0.5000, 0.0000],
            ]
        )
        actual_duty_cycles = self.apply_dendrites.duty_cycles
        all_equal = (actual_duty_cycles == expected_duty_cycles).all()
        self.assertTrue(all_equal)

        # -----------
        # 3rd Forward
        # -----------

        # Pseudo output from a Dendrite Segments module.
        dendrite_activations = torch.tensor(
            [
                [
                    [0.7358, 0.0697, 0.2366, 0.5241],
                    [0.7539, 0.9194, 0.9102, 0.9367],
                    [0.5127, 0.9793, 0.3772, 0.3956],
                ],
                [
                    [0.0988, 0.5655, 0.3606, 0.4948],
                    [0.9216, 0.6567, 0.3888, 0.7455],
                    [0.5360, 0.8731, 0.0628, 0.0343],
                ],
            ]
        )
        # Pseudo output to be contextually modulated via dendrites.
        y = torch.tensor([[0.3009, 0.5920, 0.5860], [0.9896, 0.9888, 0.4689]])

        # Modulate y via dendrites with boosting enabled.
        y_modulated, winning_indices = self.apply_dendrites(y, dendrite_activations)

        # Validate duty cycles.
        expected_duty_cycles = torch.tensor(
            [
                [0.3750, 0.3750, 0.1250, 0.1250],
                [0.1250, 0.1250, 0.2500, 0.5000],
                [0.1250, 0.6250, 0.2500, 0.0000],
            ]
        )
        actual_duty_cycles = self.apply_dendrites.duty_cycles
        all_equal = (actual_duty_cycles == expected_duty_cycles).all()
        self.assertTrue(all_equal)

    def test_backward_pass(self):

        # Pseudo context input.
        context = torch.tensor(
            [[0.8858, 0.8328, 0.3534], [0.4397, 0.5681, 0.3052]], requires_grad=True
        )

        # Compute activations of Dendrite Segments module.
        dendrite_activations = self.dendrite_segments(context)

        # Pseudo output to be contextually modulated via dendrites.
        y = torch.tensor(
            [[0.5737, 0.0753, 0.0143], [0.5244, 0.7850, 0.5987]], requires_grad=True
        )

        # Modulate output via boosted activations.
        y_modulated, winning_indices = self.apply_dendrites(y, dendrite_activations)

        # Validate winning segments.
        expected_indices = torch.tensor([[1, 2, 2], [2, 2, 2]])
        all_equal = (winning_indices == expected_indices).all()
        self.assertTrue(all_equal)

        # Run backward pass.
        y_modulated.sum().backward()

        # Validate zero gradients.
        segments = [0, 3]
        unit_0_zero_grads = self.dendrite_segments.weights.grad[0, segments, :]
        self.assertTrue((unit_0_zero_grads == 0).all())

        segments = [0, 1, 3]
        unit_1_zero_grads = self.dendrite_segments.weights.grad[1, segments, :]
        self.assertTrue((unit_1_zero_grads == 0).all())

        segments = [0, 1, 3]
        unit_2_zero_grads = self.dendrite_segments.weights.grad[2, segments, :]
        self.assertTrue((unit_2_zero_grads == 0).all())

        # Validate all gradients; this is partially redundant with the last test but
        # with a lower precision.
        expected_grad = torch.tensor(
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.1175, 0.1105, 0.0469],
                    [0.0557, 0.0720, 0.0387],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                    [0.0965, 0.1196, 0.0627],
                    [0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000],
                    [0.0669, 0.0854, 0.0455],
                    [0.0000, 0.0000, 0.0000],
                ],
            ]
        )
        grad = self.dendrite_segments.weights.grad
        all_close = torch.allclose(grad, expected_grad, atol=1e-4)
        self.assertTrue(all_close)


if __name__ == "__main__":
    unittest.main(verbosity=2)
