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

from nupic.research.frameworks.vernon import SelfSupervisedExperiment

class AutoEncoder(torch.nn.Module):
    """Quadratic layer: Computes W^T W x"""
    def __init__(self, input_dim = 784, hidden_dim = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)


    def forward(self, x):
        encoded = self.fc1(x)
        decoded = self.fc2(encoded)
        return encoded, decoded


self_supervised_config = dict(
    experiment_class=SelfSupervisedExperiment,
    num_classes=10,
    # Dataset
    dataset_class=MNIST(
        root="~/nta/data/MNIST"
    ),
    # Number of epochs
    epochs=4,
    epochs_to_validate=[3],
    batch_size=32,
    # Model class. Must inherit from "torch.nn.Module"
    model_class=AutoEncoder,
    # model model class arguments passed to the constructor
    model_args=dict(),
    optimizer_class = torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
    # Suppress logging.
    log_level="NOTSET",
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




if __name__ == "__main__":
    unittest.main(verbosity=2)
