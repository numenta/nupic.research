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

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from nupic.research.frameworks.dynamic_sparse import (
    global_add_by_abs_grad,
    global_prune_by_abs_weight,
)
from nupic.torch.modules import SparseWeights, rezero_weights

# Chosen so that all linear transformations can be of shape 4 x 4
INPUT_SHAPE = (4,)


class SimpleSparseMLP(torch.nn.Module):
    def __init__(self, input_shape=INPUT_SHAPE, hidden_size=4, output_size=4):
        super().__init__()

        input_size = np.prod(input_shape)
        self.flatten = torch.nn.Flatten()
        self.lin1 = SparseWeights(
            torch.nn.Linear(input_size, hidden_size, bias=None), sparsity=0.5
        )
        self.lin2 = SparseWeights(
            torch.nn.Linear(hidden_size, hidden_size, bias=None), sparsity=0.5
        )
        self.lin3 = SparseWeights(
            torch.nn.Linear(hidden_size, output_size, bias=None), sparsity=0.5
        )

    def forward(self, x):
        y = self.flatten(x)
        y = self.lin1(y)
        y = self.lin2(y)
        y = self.lin3(y)
        return y


def all_equal(tensor1, tensor2):
    return (tensor1 == tensor2).all()


def all_close(tensor1, tensor2, atol=1e-4):
    return torch.allclose(tensor1, tensor2, atol=atol)


class GlobalPruningTest(unittest.TestCase):
    def setUp(self):
        """
        Initialize model weights and mask.
        """
        model = SimpleSparseMLP()

        model.lin1.module.weight.data[:] = torch.tensor(
            [
                [0.00, 0.34, 0.00, -0.45],
                [-0.37, 0.00, 0.00, -0.45],
                [-0.31, 0.02, 0.00, 0.00],
                [0.42, -0.15, 0.00, 0.00],
            ],
            requires_grad=True,
        )
        self.initial_lin1_mask = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ).half()
        model.lin1.zero_mask[:] = self.initial_lin1_mask

        model.lin2.module.weight.data[:] = torch.tensor(
            [
                [-0.09, 0.22, 0.00, 0.00],
                [0.13, 0.28, 0.00, 0.00],
                [0.00, 0.24, 0.00, -0.16],
                [0.00, 0.00, 0.01, 0.02],
            ],
            requires_grad=True,
        )
        self.initial_lin2_mask = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        ).half()
        model.lin2.zero_mask[:] = self.initial_lin2_mask

        model.lin3.module.weight.data[:] = torch.tensor(
            [
                [-0.14, 0.00, 0.00, -0.08],
                [0.00, -0.36, 0.00, -0.43],
                [0.00, 0.33, 0.00, -0.50],
                [0.22, 0.00, 0.41, 0.00],
            ],
            requires_grad=True,
        )
        self.initial_lin3_mask = torch.tensor(
            [
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        ).half()
        model.lin3.zero_mask[:] = self.initial_lin3_mask

        self.model = model
        self.model.apply(rezero_weights)

    def test_global_pruning_and_regrowing(self):

        # ----------
        # Prune
        # ----------

        sparse_modules = [self.model.lin1, self.model.lin2, self.model.lin3]
        global_prune_by_abs_weight(sparse_modules, prune_fraction=0.5)
        self.model.apply(rezero_weights)

        # Validate pruned weights
        expected_w1 = torch.tensor(
            [
                [0.0000, 0.3400, 0.0000, -0.4500],
                [-0.3700, 0.0000, 0.0000, -0.4500],
                [-0.3100, 0.0000, 0.0000, 0.0000],
                [0.4200, 0.0000, 0.0000, 0.0000],
            ]
        )
        self.assertTrue(all_equal(self.model.lin1.weight, expected_w1))

        expected_w2 = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.2800, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
            ]
        )
        self.assertTrue(all_equal(self.model.lin2.weight, expected_w2))

        expected_w3 = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, -0.3600, 0.0000, -0.4300],
                [0.0000, 0.3300, 0.0000, -0.5000],
                [0.0000, 0.0000, 0.4100, 0.0000],
            ]
        )
        self.assertTrue(all_equal(self.model.lin3.weight, expected_w3))

        # Validate pruned mask is a subset of the original mask.
        pruned_lin1_mask = self.model.lin1.zero_mask
        pruned_lin2_mask = self.model.lin2.zero_mask
        pruned_lin3_mask = self.model.lin3.zero_mask
        self.assertTrue((pruned_lin1_mask >= self.initial_lin1_mask).all())
        self.assertTrue((pruned_lin2_mask >= self.initial_lin2_mask).all())
        self.assertTrue((pruned_lin3_mask >= self.initial_lin3_mask).all())

        # Validate number of off weights.
        zero_masks = parameters_to_vector(self.model.buffers())
        weights = parameters_to_vector(self.model.parameters())
        self.assertEqual((weights == 0).sum(), 36)
        self.assertEqual(zero_masks.sum(), 36)

        # ----------
        # Regrow
        # ----------

        # Pseudo forward pass to accumulate gradients.
        x = torch.tensor(
            [[0.35, 0.94, 0.10, 0.31], [0.05, 0.16, 0.46, 0.11]], requires_grad=True
        )
        self.model(x).sum().backward()

        # Regrow weights per the largest abs gradients.
        global_add_by_abs_grad(sparse_modules, num_add=12)
        self.model.apply(rezero_weights)

        # Validate regrown weights
        expected_w1 = torch.tensor(
            [
                [0.0000, 0.3400, 0.0000, -0.4500],
                [-0.3700, 0.0000, 0.0000, -0.4500],
                [-0.3100, 0.0000, 0.0000, 0.0000],
                [0.4200, 0.0000, 0.0000, 0.0000],
            ]
        )
        self.assertTrue(all_close(self.model.lin1.weight, expected_w1))

        expected_w2 = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.2800, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000],
            ]
        )
        self.assertTrue(all_close(self.model.lin2.weight, expected_w2))

        expected_w3 = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, -0.3600, 0.0000, -0.4300],
                [0.0000, 0.3300, 0.0000, -0.5000],
                [0.0000, 0.0000, 0.4100, 0.0000],
            ]
        )
        self.assertTrue(all_close(self.model.lin3.weight, expected_w3))

        # Validate regrown mask is a subset of the pruned mask.
        self.assertTrue((self.model.lin1.zero_mask <= pruned_lin1_mask).all())
        self.assertTrue((self.model.lin2.zero_mask <= pruned_lin2_mask).all())
        self.assertTrue((self.model.lin3.zero_mask <= pruned_lin3_mask).all())

        # Validate number of off weights.
        zero_masks = parameters_to_vector(self.model.buffers())
        weights = parameters_to_vector(self.model.parameters())
        self.assertEqual((weights == 0).sum(), 36)
        self.assertEqual(zero_masks.sum(), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)
