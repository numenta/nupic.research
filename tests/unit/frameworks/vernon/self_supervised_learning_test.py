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
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from nupic.research.frameworks.vernon import SelfSupervisedExperiment

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        encoded = self.fc1(x)
        decoded = self.fc2(encoded)
        return decoded.view(-1, 1, 28, 28)

    def encode(self, x):
        x = x.flatten(start_dim=1)
        encoded = self.fc1(x)
        return encoded


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim = 20, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

MEAN = 0.13062755
STDEV = 0.30810780
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN,
                                                                           STDEV)])


self_supervised_config = dict(
    experiment_class=SelfSupervisedExperiment,
    num_classes=10,
    # Dataset
    dataset_class=MNIST,
    dataset_args=dict(
        unsupervised=dict(root="~/nta/data/MNIST", train=True, transform=transform),
        supervised=dict(root="~/nta/data/MNIST", train=True, transform=transform),
        validation=dict(root="~/nta/data/MNIST", train=False, transform=transform)
    ),
    # Number of epochs
    epochs=5,
    epochs_to_validate=[2, 4],
    supervised_training_epochs_per_validation = 1,
    batch_size=32,
    batch_size_supervised=32,


    # Model class. Must inherit from "torch.nn.Module"
    model_class=AutoEncoder,
    # model model class arguments passed to the constructor
    model_args=dict(),

    classifier_model_class=LinearClassifier,
    classifier_model_args=dict(),

    optimizer_class = torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.001),

    classifier_optimizer_class=torch.optim.SGD,
    classifier_optimizer_args=dict(lr=0.001),

    loss_function=torch.nn.functional.mse_loss,
    loss_function_supervised=torch.nn.functional.cross_entropy,
)


class SelfSupervisedLearningTest(unittest.TestCase):
    """
    This is a test class for the `GradientMetrics` mixin.
    """
    def test_gradient_metrics_supervised_experiment(self):
        """
        Test whether GradientMetrics tracking and plotting works in the supervised
        setting.
        """
        # Setup experiment and initialize model.
        exp = self_supervised_config["experiment_class"]()
        exp.setup_experiment(self_supervised_config)
        # Loop through some pseudo epochs.
        for i in range(5):
            ret = exp.run_epoch()
            print(ret)


if __name__ == "__main__":
    unittest.main(verbosity=2)
