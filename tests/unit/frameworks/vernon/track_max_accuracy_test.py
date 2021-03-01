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
# %%
import unittest

import numpy as np
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.vernon import SupervisedExperiment, mixins


class TrackMaxAccuracySupervisedExperiment(mixins.TrackMaxAccuracy,
                                           SupervisedExperiment):
    pass


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        in_features = np.prod(input_shape)
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(in_features, num_classes, bias=False)

    def forward(self, x):
        y = self.flatten(x)
        return self.classifier(y)


def fake_data(size=100, image_size=(1, 4, 4), train=False):
    return FakeData(size=size, image_size=image_size, transform=ToTensor())


simple_supervised_config = dict(
    experiment_class=TrackMaxAccuracySupervisedExperiment,
    num_classes=10,

    # Dataset
    dataset_class=fake_data,

    # Number of epochs
    epochs=5,
    batch_size=5,
    epochs_to_validate=range(5),

    # Model class. Must inherit from "torch.nn.Module"
    model_class=SimpleMLP,
    # model model class arguments passed to the constructor
    model_args=dict(
        num_classes=10,
        input_shape=(1, 4, 4),
    ),

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),

    # Suppress logging.
    log_level="NOTSET"
)


class TrackMaxAccuracyUnitTest(unittest.TestCase):
    """
    This is a test class for the `TrackMaxAccuracy` mixin.
    """

    def test_track_max_accuracy_supervised_experiment(self):
        """
        Test whether TrackMaxAccuracy works in the supervised setting.
        """

        # Setup experiment and initialize model.
        exp = simple_supervised_config["experiment_class"]()
        exp.setup_experiment(simple_supervised_config)

        # Loop through some pseudo epochs
        curr_max = 0
        for _ in range(5):
            ret = exp.run_epoch()
            self.assertTrue("max_accuracy" in ret)
            self.assertTrue(ret["max_accuracy"] >= curr_max)
            curr_max = ret["max_accuracy"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
