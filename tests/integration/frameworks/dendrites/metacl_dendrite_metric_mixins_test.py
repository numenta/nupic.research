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

import numpy as np
import pytest
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.torch.modules import KWinners, SparseWeights

dendrites = pytest.importorskip("nupic.research.frameworks.dendrites")
dendrites_mixins = pytest.importorskip("nupic.research.frameworks.dendrites.mixins")
metacl = pytest.importorskip(
    "nupic.research.frameworks.meta_continual_learning.experiments")


class TrackStatsMetaCLExperiment(dendrites_mixins.PlotDendriteMetrics,
                                 metacl.MetaContinualLearningExperiment):
    pass


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        in_features = np.prod(input_shape)
        self.dendritic_gate = dendrites.DendriticAbsoluteMaxGate1d()
        self.flatten = torch.nn.Flatten()
        self.kwinners = KWinners(n=16, percent_on=0.75, k_inference_factor=1)
        self.classifier = SparseWeights(
            torch.nn.Linear(in_features, num_classes, bias=False),
            sparsity=0.5,
        )

    def forward(self, x):

        y = self.flatten(x)

        batch_size, num_units = y.shape
        num_segments = 10
        dendritic_activations = torch.rand(batch_size, num_units, num_segments)
        y = self.dendritic_gate(y, dendritic_activations).values

        y = self.kwinners(y)
        return self.classifier(y)


def fake_data(size=100, image_size=(1, 4, 4), train=False):
    return FakeData(size=size, image_size=image_size, transform=ToTensor())


simple_metacl_config = dict(

    experiment_class=TrackStatsMetaCLExperiment,
    num_classes=10,
    
    # FIXME SKIP GPU FOR NOW, 
    # eventually we must test it on GPU too
    device="cpu",
    
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

    plot_dendrite_metrics_args=dict(
        include_modules=[dendrites.DendriticAbsoluteMaxGate1d],
        mean_selected=dict(
            max_samples_to_plot=400,
            plot_freq=2,
            plot_func=dendrites.plot_mean_selected_activations
        )
    ),

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
    fast_params=[".*"],  # <- all params get updated in inner loop

    # Suppress logging.
    log_level="NOTSET",
)


class PlotDendriteMetricsTest(unittest.TestCase):
    """
    This is a test class for the `PlotDendriteMetrics` mixin.
    """

    def test_dendrite_metrics_tracking_metacl_experiment(self):
        """
        Test whether TrackMeanSelectedActivations works in the metacl setting.
        """

        # Setup experiment and initialize model.
        exp = simple_metacl_config["experiment_class"]()
        exp.setup_experiment(simple_metacl_config)

        # Loop through some pseudo epochs.
        for i in range(5):
            ret = exp.run_epoch()

            # The plot frequency is 2 and should be logged every 2 epochs.
            if i % 2 == 0:
                self.assertTrue("mean_selected/dendritic_gate" in ret)

                # Raw data should be logged whenever a plot is logged.
                self.assertTrue("targets/dendritic_gate" in ret)
                self.assertTrue("dendrite_activations/dendritic_gate" in ret)
                self.assertTrue("winning_mask/dendritic_gate" in ret)

                # The raw data should be a numpy array.
                targets = ret["targets/dendritic_gate"]
                activations = ret["dendrite_activations/dendritic_gate"]
                winners = ret["winning_mask/dendritic_gate"]

                self.assertTrue(isinstance(targets, np.ndarray))
                self.assertTrue(isinstance(activations, np.ndarray))
                self.assertTrue(isinstance(winners, np.ndarray))


if __name__ == "__main__":
    unittest.main(verbosity=2)
