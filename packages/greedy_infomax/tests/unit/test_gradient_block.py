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
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.greedy_infomax.models.utility_layers import GradientBlock


class TestGradientBlock(unittest.TestCase):

    def test_gradients_blocked(self):
        """
        Test that gradients are blocked preceding the GradientBlock module.
        """
        data = FakeData(size=10, image_size=(1, 10, 10), num_classes=10,
                        transform=ToTensor())
        dataloader = DataLoader(data, batch_size=10)
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
            GradientBlock(),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        optimizer = Adam(model.parameters(), lr=1e-4)
        for data, targets in dataloader:
            out = model(data)
            error_loss = torch.nn.functional.cross_entropy(out, targets)
            optimizer.zero_grad()
            error_loss.backward()
            conv1_grads = model._modules["0"].weight.grad
            linear_grads = model._modules["3"].weight.grad
            self.assertIsNone(conv1_grads)
            self.assertIsNotNone(linear_grads)

    def test_gradients_not_blocked(self):
        """
        Test that gradients are not blocked following the GradientBlock module.
        """
        data = FakeData(size=10, image_size=(1, 10, 10), num_classes=10,
                        transform=ToTensor())
        dataloader = DataLoader(data, batch_size=10)
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        optimizer = Adam(model.parameters(), lr=1e-4)
        for data, targets in dataloader:
            out = model(data)
            error_loss = torch.nn.functional.cross_entropy(out, targets)
            optimizer.zero_grad()
            error_loss.backward()
            conv1_grads = model._modules["0"].weight.grad
            linear_grads = model._modules["2"].weight.grad
            self.assertIsNotNone(conv1_grads)
            self.assertIsNotNone(linear_grads)


if __name__ == "__main__":
    unittest.main()
