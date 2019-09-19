#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import torch.nn

from nupic.research.frameworks.pytorch.models.le_sparse_net import LeSparseNet


class LeSparseNetTest(unittest.TestCase):

    def test_default(self):
        """
        Make sure we get something reasonable with default parameters and
        that it runs.
        """
        model = LeSparseNet()
        x = torch.randn((2,) + (1, 32, 32))
        self.assertGreater(len(model._modules), 2)
        y = model(x)
        self.assertEqual(y.size()[0], 2)
        self.assertEqual(y.size()[1], 10)

    def test_no_cnn(self):
        """Create a net where there are no CNN blocks."""
        model = LeSparseNet(
            cnn_out_channels=(),
            linear_n=(100, 200),
            linear_activity_percent_on=(0.1, 1.0),
            linear_weight_percent_on=(1.0, 0.4),
        )
        self.assertGreater(len(model._modules), 2)
        for key in model._modules.keys():
            self.assertFalse("cnn" in key)

        # Run some input through it and ensure it doesn't crash
        x = torch.randn((2,) + (1, 32, 32))
        y = model(x)
        self.assertEqual(y.size()[0], 2)
        self.assertEqual(y.size()[1], 10)

    def test_no_linear(self):
        """Create a net where there are no linear blocks."""
        model = LeSparseNet(
            cnn_out_channels=(8, ),
            linear_n=(),
        )
        self.assertGreater(len(model._modules), 2)
        for key in model._modules.keys():
            self.assertFalse("linear" in key)

        # Run some input through it and ensure it doesn't crash
        x = torch.randn((2,) + (1, 32, 32))
        y = model(x)
        self.assertEqual(y.size()[0], 2)
        self.assertEqual(y.size()[1], 10)

    def test_irregular(self):
        """Create a net where different blocks have different sparsities."""
        model = LeSparseNet(
            cnn_out_channels=(8, 8),
            cnn_activity_percent_on=(0.1, 0.2),
            cnn_weight_percent_on=(1.0, 0.2),
            linear_n=(100, 200),
            linear_activity_percent_on=(0.1, 1.0),
            linear_weight_percent_on=(1.0, 0.4),
        )
        self.assertGreater(len(model._modules), 2)

        # Run some input through it and ensure it doesn't crash
        x = torch.randn((2,) + (1, 32, 32))
        y = model(x)
        self.assertEqual(y.size()[0], 2)
        self.assertEqual(y.size()[1], 10)


if __name__ == "__main__":
    unittest.main()
