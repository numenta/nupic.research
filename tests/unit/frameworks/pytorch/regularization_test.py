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

import copy
import unittest

import torch

from nupic.research.frameworks.pytorch import l1_regularization_step


class L1RegulatizationStepTest(unittest.TestCase):

    def test_all_zeros(self):
        """
        The zero vector should not be modified
        """
        weight = torch.zeros((10, 10))
        weight = torch.nn.Parameter(weight)

        bias = torch.zeros((10,))
        bias = torch.nn.Parameter(bias)

        l1_regularization_step(params=[weight, bias], lr=0.1, weight_decay=0.1)

        self.assertTrue((weight == 0.0).all().item())
        self.assertTrue((bias == 0.0).all().item())

    def test_non_zero_parameter(self):
        """
        The updated parameters should match the hand-derived solution
        """
        weight = torch.Tensor([
            [-0.4884, -0.2566, 0.3548],
            [0.2883, -0.5463, 0.0184],
            [0.2392, 0.0000, 0.3523]
        ])
        weight = torch.nn.Parameter(weight)

        bias = torch.Tensor([-0.4757, -0.4825, -0.0000])
        bias = torch.nn.Parameter(bias)

        l1_regularization_step(params=[weight, bias], lr=0.15, weight_decay=0.25)

        expected_weight = torch.Tensor([
            [-0.4509, -0.2191, 0.3173],
            [0.2508, -0.5088, -0.0191],
            [0.2017, 0.0000, 0.3148]
        ])

        expected_bias = torch.Tensor([-0.4382, -0.4450, 0.0000])

        self.assertTrue(torch.allclose(weight, expected_weight))
        self.assertTrue(torch.allclose(bias, expected_bias))

    def test_zero_weight_decay(self):
        """
        No parameters should not be modified if `weight_decay` is set to zero
        """
        weight = torch.randn((7, 7))
        weight = torch.nn.Parameter(weight)

        bias = torch.randn((7,))
        bias = torch.nn.Parameter(bias)

        # Make copy of original parameters before update
        weight_original = copy.deepcopy(weight)
        bias_original = copy.deepcopy(bias)

        l1_regularization_step(params=[weight, bias], lr=0.1, weight_decay=0.0)

        self.assertTrue(torch.allclose(weight, weight_original))
        self.assertTrue(torch.allclose(bias, bias_original))

    def test_requires_grad(self):
        """
        Any parameters whose `requires_grad` attribute is False should not be modified
        """
        weight = torch.randn((3, 11))
        weight = torch.nn.Parameter(weight)

        bias = torch.randn((11,))
        bias = torch.nn.Parameter(bias, requires_grad=False)

        # Make copy of original parameters before update
        weight_original = copy.deepcopy(weight)
        bias_original = copy.deepcopy(bias)

        l1_regularization_step(params=[weight, bias], lr=0.1, weight_decay=0.1)

        # Here, we assert that at least 1 weight has changed and the bias remains fixed
        self.assertFalse(torch.allclose(weight, weight_original))
        self.assertTrue(torch.allclose(bias, bias_original))


if __name__ == "__main__":
    unittest.main()
