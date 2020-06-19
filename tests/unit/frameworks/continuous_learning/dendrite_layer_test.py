#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import unittest

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

from nupic.research.frameworks.continuous_learning.dend_kwinners import (
    DendriteKWinners2d,
    DendriteKWinners2dLocal,
)
from nupic.research.frameworks.continuous_learning.dendrite_layers import DendriteOutput


class DendKWinnerTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(3, 2, 4, 1)

    def test_dend_kwinner_max(self):
        """
        Test if the activation returns the correct units
        """
        x = self.x
        b, c, h, w = x.shape
        f = DendriteKWinners2dLocal

        result = f.apply(x, 1)

        expected_ = x.reshape(b * c, h)
        # these should be (absolute) the k=1 values
        expected = expected_.max(dim=1).values.abs()

        result_ = result.reshape(b * c, h)
        result_ = result_.abs().max(dim=1).values

        num_correct = (result_ == expected).sum()
        self.assertEqual(num_correct, expected.shape[0])

    def test_dend_kwinner_grad_values(self):
        """
        Test gradients using pytorch.autograd.gradcheck
        """
        x = self.x.requires_grad_(True).double()  # needs to be double precision
        b, c, h, w = x.shape

        kw = DendriteKWinners2d(c, k=1)  # test needs module, not function

        self.assertTrue(gradcheck(kw, x))

    def test_dend_output_grad_inds(self):
        """
        Make sure the gradients fall where they have to
        """

        dend_output = DendriteOutput(out_dim=10, dendrites_per_unit=3)

        x = torch.randn(8, 30)
        target = torch.rand(8,).long()

        loss_fn = F.cross_entropy

        loss = loss_fn(dend_output(x), target)
        loss.backward()

        w = dend_output.weight.grad.detach()
        w[w.abs() > 0] = 1.0  # set all gradients to 1

        # sum over gradient rows and check it's == 30
        sum_ = 0
        for k in range(w.shape[0]):
            sum_ += w[k, :].sum()

        self.assertEqual(sum_, 30)


if __name__ == "__main__":
    unittest.main()
