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

import os
import unittest

import numpy as np
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.dendrites import (
    DendriticAbsoluteMaxGate1d,
    plot_winning_segment_distributions,
)
from nupic.research.frameworks.vernon import MetaContinualLearningExperiment, mixins
from nupic.torch.modules import KWinners, SparseWeights


class TrackedSegmentsMetaCLExperiment(mixins.PlotDendriteMetrics,
                                      MetaContinualLearningExperiment):
    pass


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        in_features = np.prod(input_shape)
        self.dendritic_gate = DendriticAbsoluteMaxGate1d()
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


def get_plot_args():
    return dict(
        num_units_to_plot=5,
        seed=torch.initial_seed()
    )


def plot_winning_segment_distributions_(
    dendrite_activations_,
    winning_mask,
    targets_,
    **kwargs
):
    """Adjust signature to work with `PlotDendriteMetrics` mixin."""
    return plot_winning_segment_distributions(winning_mask, **kwargs)


simple_metacl_config = dict(

    experiment_class=TrackedSegmentsMetaCLExperiment,
    num_classes=10,
    fast_params=[".*"],  # <- all params

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

    # Plotting args.
    plot_dendrite_metrics_args=dict(
        include_modules=[DendriticAbsoluteMaxGate1d],
        winning_segments=dict(
            plot_func=plot_winning_segment_distributions,
            plot_freq=2,
            plot_args=get_plot_args,
            max_samples_to_track=10000,
        )
    ),

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),

    # Suppress logging.
    log_level="NOTSET"
)


class CustomDendritMetricsTest(unittest.TestCase):
    """
    This is a test class for the `PlotDendriteMetrics` mixin used for plotting custom
    metrics.
    """

    def test_plot_winning_segment_distributions(self):

        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        # Setup experiment and initialize model.
        exp = simple_metacl_config["experiment_class"]()
        exp.setup_experiment(simple_metacl_config)

        # Loop through some pseudo epochs.
        for i in range(6):
            ret = exp.run_epoch()

            if i % 2 == 0:
                # import matplotlib.pyplot as plt
                # plt.show()
                self.assertTrue("winning_segments/dendritic_gate" in ret)
            else:
                self.assertTrue("winning_segments/dendritic_gate" not in ret)


if __name__ == "__main__":
    unittest.main(verbosity=2)
