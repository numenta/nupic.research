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
import torch.nn as nn
import torch.nn.functional as F

from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.remove_batchnorm import remove_batchnorm
from nupic.torch.modules import Flatten


class SimpleCNN(nn.Sequential):
    """
    Simple CNN model for testing batch norm removal. One CNN layer plus
    one fully connected layer plus a linear output layer.
    """

    def __init__(self,
                 cnn_out_channels=(2, 2),
                 linear_units=3,
                 ):
        super(SimpleCNN, self).__init__()
        # input_shape = (1, 32, 32)
        # First Sparse CNN layer
        self.add_module("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5))
        self.add_module("cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0],
                                                         affine=False))
        self.add_module("cnn1_maxpool", nn.MaxPool2d(2))
        self.add_module("cnn1_relu", nn.ReLU())

        self.add_module("flatten", Flatten())

        # Sparse Linear layer
        self.add_module("linear", nn.Linear(196 * cnn_out_channels[0], linear_units))
        self.add_module("linear_bn", nn.BatchNorm1d(linear_units, affine=False))
        self.add_module("linear_relu", nn.ReLU())

        # Classifier
        self.add_module("output", nn.Linear(linear_units, 12))


def train_randomly(model, num_samples=20):
    """
    Train the model on random inputs to ensure the batchnorm really learns something.
    """
    # Create a random training set
    x = torch.randn((num_samples,) + (1, 32, 32))
    targets = torch.randint(0, 12, (num_samples,))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.2)
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()
        # print(loss.item())


class RemoveBatchnormTest(unittest.TestCase):

    def test_simple_cnn(self):
        """Compare a network with itself after batchnorm is removed."""
        model = SimpleCNN()
        train_randomly(model)
        model2 = remove_batchnorm(model)

        self.assertLess(len(model2._modules.keys()), len(model._modules.keys()))
        self.assertTrue(compare_models(model, model2, (1, 32, 32)))


if __name__ == "__main__":
    unittest.main()
