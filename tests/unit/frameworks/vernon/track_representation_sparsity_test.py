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
from copy import deepcopy

import numpy as np
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.pytorch.hooks import ModelHookManager
from nupic.research.frameworks.vernon import SupervisedExperiment, mixins
from nupic.torch.modules import KWinners, SparseWeights


class TrackStatsSupervisedExperiment(mixins.TrackRepresentationSparsity,
                                     SupervisedExperiment):
    pass


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        in_features = np.prod(input_shape)
        self.flatten = torch.nn.Flatten()
        self.kwinners = KWinners(n=16, percent_on=0.75, k_inference_factor=1)
        self.classifier = SparseWeights(
            torch.nn.Linear(in_features, num_classes, bias=False),
            sparsity=0.5,
        )

    def forward(self, x):
        y = self.flatten(x)
        y = self.kwinners(y)
        return self.classifier(y)


def fake_data(size=100, image_size=(1, 4, 4), train=False):
    return FakeData(size=size, image_size=image_size, transform=ToTensor())


simple_supervised_config = dict(

    experiment_class=TrackStatsSupervisedExperiment,
    num_classes=10,

    # Dataset
    dataset_class=fake_data,

    # Number of epochs
    epochs=1,
    batch_size=5,

    # Model class. Must inherit from "torch.nn.Module"
    model_class=SimpleMLP,
    # model model class arguments passed to the constructor
    model_args=dict(
        num_classes=10,
        input_shape=(1, 4, 4),
    ),

    track_input_sparsity_args=dict(
        include_modules=[KWinners, torch.nn.Linear]
    ),
    track_output_sparsity_args=dict(
        include_modules=[torch.nn.Linear]
    ),

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
    fast_params=[".*"],  # <- all params get updated in inner loop

    # Suppress logging.
    log_level="NOTSET"
)


class TrackRepresentationSparsityTest(unittest.TestCase):
    """
    This is a test class for the `TrackRepresentationSparsity` mixin.
    """

    def test_sparsity_tracking_supervised_experiment(self):
        """
        Ensure both input and output sparsities can be tracked during a supervised
        experiment.
        """

        # Setup experiment and initialize model.
        exp = simple_supervised_config["experiment_class"]()
        exp.setup_experiment(simple_supervised_config)

        # Validate that the hook managers are not null.
        self.assertIsInstance(exp.input_hook_manager, ModelHookManager)
        self.assertIsInstance(exp.output_hook_manager, ModelHookManager)

        # Loop through some pseudo epochs.
        for _ in range(5):
            ret = exp.run_epoch()

            # Validate the expected stats have been tracked.
            name = "input_sparsity/kwinners (KWinners)"
            self.assertTrue(name in ret.keys())
            name = "input_sparsity/classifier.module (Linear)"
            self.assertTrue(name in ret.keys())
            sparsity = ret[name]
            self.assertTrue(np.isclose(sparsity, 0.25, atol=1e-3))

            name = "output_sparsity/classifier.module (Linear)"
            self.assertTrue(name in ret.keys())
            sparsity = ret[name]
            self.assertTrue(np.isclose(sparsity, 0, atol=1e-3))

    def test_no_tracking_args_given_supervised_experiment(self):
        """
        Test the edge case for supervised experiment where neither input and output
        sparsities are tracked.
        """

        # Remove tracking params.
        no_tracking_config = deepcopy(simple_supervised_config)
        no_tracking_config.pop("track_input_sparsity_args")
        no_tracking_config.pop("track_output_sparsity_args")

        # Setup experiment and initialize model.
        exp = no_tracking_config["experiment_class"]()
        with self.assertLogs("TrackStatsSupervisedExperiment", "WARNING"):
            exp.setup_experiment(no_tracking_config)

        # Validate that the hook managers are null.
        self.assertEqual(len(exp.input_hook_manager.hooks), 0)
        self.assertEqual(len(exp.output_hook_manager.hooks), 0)

        # Loop through some pseudo epochs.
        for _ in range(5):
            ret = exp.run_epoch()

            # Validate that nothing is being tracked.
            for k in ret.keys():
                self.assertTrue("input_sparsity" not in k)
                self.assertTrue("output_sparsity" not in k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
