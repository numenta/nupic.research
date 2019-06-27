# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

import nupic.research.frameworks.pytorch.functions as F


class TestContext(object):
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, x):
        self.saved_tensors = (x,)


class KWinnersTest(unittest.TestCase):
    """"""

    def setUp(self):
        # Tests will use 3-dim tensor (batch_size, width, height)

        # Batch size 1 (2 columns, 4 cells)
        x = torch.ones((1, 2, 4))
        x[0, 0, 1] = 1.2
        x[0, 0, 3] = 1.2
        x[0, 1, 0] = 1.2
        x[0, 1, 2] = 1.3
        self.x = x
        self.gradient = torch.rand(x.shape)

        # Batch size 2
        x = torch.ones((2, 5, 2))
        x[0, 0, 1] = 1.1
        x[0, 1, 0] = 1.2
        x[0, 2, 1] = 1.3

        x[1, 0, 0] = 1.4
        x[1, 1, 0] = 1.5
        x[1, 2, 1] = 1.7
        self.x2 = x
        self.gradient2 = torch.rand(x.shape)

    def test_one(self):
        """Equal duty cycle, boost factor 0, k=4, batch size 1."""
        x = self.x

        ctx = TestContext()

        result = F.KWinnersMask.forward(ctx, x, k=2)

        expected = torch.zeros_like(x)
        expected[0, 0, 1] = 1
        expected[0, 0, 3] = 1
        expected[0, 1, 0] = 1
        expected[0, 1, 2] = 1

        self.assertEqual(result.shape, expected.shape)

        num_correct = (result == expected).sum()
        self.assertEqual(num_correct, result.reshape(-1).size()[0])

        indices = ctx.saved_tensors[0]
        expected_indices = torch.tensor([[1, 3], [2, 0]])
        num_correct = (indices == expected_indices).sum()
        self.assertEqual(num_correct, 4)

        # Test that gradient values are in the right places, that their sum is
        # equal, and that they have exactly the right number of nonzeros
        grad_x, _, _, _ = F.KWinnersMask.backward(ctx, self.gradient)

        # TODO: Better gradient test

        self.assertEqual(len(grad_x.nonzero()), 4)


if __name__ == "__main__":
    unittest.main()
