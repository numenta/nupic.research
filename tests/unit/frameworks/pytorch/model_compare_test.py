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

import copy
import unittest

import torch
import torch.nn

from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.torch.modules import Flatten


def simple_linear_net():
    return torch.nn.Sequential(
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2)
    )


def simple_conv_net():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        Flatten(),
        torch.nn.Linear(588, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2)
    )


class ModelCompareTest(unittest.TestCase):

    def test_identical(self):
        """Compare a network with itself"""
        model = simple_linear_net()
        self.assertTrue(compare_models(model, model, (32,)))

    def test_almost_identical(self):
        """Compare a network with itself except for one weight"""
        model1 = simple_linear_net()
        model2 = copy.deepcopy(model1)
        model1._modules["0"].weight[0][0] = 1.0
        model2._modules["0"].weight[0][0] = -1.0
        self.assertFalse(compare_models(model1, model2, (32,)))

    def test_different(self):
        """Compare two random networks"""
        model1 = simple_linear_net()
        model2 = simple_linear_net()
        self.assertFalse(compare_models(model1, model2, (32,)))

    def test_conv_identical(self):
        """Compare a conv network with itself"""
        model = simple_conv_net()
        self.assertTrue(compare_models(model, model, (1, 32, 32)))

    def test_conv_different(self):
        """Compare two random conv networks"""
        model1 = simple_conv_net()
        model2 = simple_conv_net()
        self.assertFalse(compare_models(model1, model2, (1, 32, 32)))


if __name__ == "__main__":
    unittest.main()
