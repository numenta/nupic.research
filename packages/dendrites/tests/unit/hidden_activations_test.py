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
from copy import deepcopy

import numpy as np
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.dendrites import mixins
from nupic.research.frameworks.pytorch.hooks import ModelHookManager
from nupic.research.frameworks.vernon import SupervisedExperiment
from nupic.torch.modules import KWinners, SparseWeights


class TrackStatsSupervisedExperiment(mixins.PlotHiddenActivations,
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

    plot_hidden_activations_args=dict(
        include_modules=[torch.nn.Linear, KWinners],
        plot_freq=2,
        max_samples_to_plot=400
    ),

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),

    # Suppress logging.
    log_level="NOTSET"
)


class PlotHiddenActivationsTest(unittest.TestCase):
    """
    This is a test class for the `PlotHiddenActivations` mixin.
    """

    def test_hidden_activations_tracking_supervised_experiment(self):
        """
        Test whether the mixin tracks hidden activations in the supervised setting.
        """

        # Setup experiment and initialize model.
        exp = simple_supervised_config["experiment_class"]()
        exp.setup_experiment(simple_supervised_config)

        # Validate that the hook managers are not null.
        self.assertIsInstance(exp.ha_hook, ModelHookManager)

        # Loop through some pseudo epochs.
        for i in range(6):
            ret = exp.run_epoch()

            # The plot frequency is 2 and should be logged every 2 epochs.
            if i % 2 == 0:
                self.assertTrue("hidden_activations/kwinners" in ret)
                self.assertTrue("hidden_activations/classifier.module" in ret)

                ha = ret["_activations/classifier.module"]
                self.assertTrue(isinstance(ha, np.ndarray))

            # All the the tensors tracked should be of the same batch size.
            batch_size1 = exp.ha_targets.size(0)
            batch_size2 = exp.ha_hook.hooks[0]._activations.shape[0]

            if i == 0:
                self.assertTrue(batch_size1 == batch_size2 == 200)
            else:
                # These should cap off at `num_samples_to_track=400`
                self.assertTrue(batch_size1 == batch_size2 == 400)

    def test_no_tracking_args_given_supervised_experiment(self):
        """
        Test the edge case where hidden activations are not tracked in the
        supervised setting.
        """

        # Remove tracking params.
        no_tracking_config = deepcopy(simple_supervised_config)
        no_tracking_config.pop("plot_hidden_activations_args")

        # Setup experiment and initialize model.
        exp = no_tracking_config["experiment_class"]()
        exp.setup_experiment(no_tracking_config)

        # Validate that the hook managers are null.
        self.assertEqual(len(exp.ha_hook.hooks), 0)

        # Loop through some pseudo epochs.
        for _ in range(5):
            ret = exp.run_epoch()

            # Validate that nothing is being tracked.
            for k in ret.keys():
                self.assertTrue("hidden_activations" not in k)


if __name__ == "__main__":
    unittest.main()
