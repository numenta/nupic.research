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

from nupic.research.frameworks.dendrites import DendriticAbsoluteMaxGate1d
from nupic.research.frameworks.vernon import (
    MetaContinualLearningExperiment,
    SupervisedExperiment,
    mixins,
)
from nupic.torch.modules import KWinners, SparseWeights


class TrackStatsSupervisedExperiment(mixins.TrackMeanSelectedActivations,
                                     SupervisedExperiment):
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

    track_mean_selected_activations_args=dict(
        include_modules=[DendriticAbsoluteMaxGate1d]
    ),

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
    fast_params=[".*"],  # <- all params get updated in inner loop

    # Suppress logging.
    log_level="NOTSET"
)


# simple_metacl_config = {**simple_supervised_config}
# simple_metacl_config.update(
#     experiment_class=TrackStatsMetaCLExperiment,
#     fast_params=[".*"],  # <- all params
# )


class TrackRepresentationSparsityTest(unittest.TestCase):
    """
    This is a test class for the `TrackRepresentationSparsityMetaCL` and
    `TrackRepresentationSparsity` mixins.
    """

    def test_sparsity_tracking_supervised_experiment(self):
        """
        Ensure both input and output sparsities can be tracked during a supervised
        experiment.
        """

        # Setup experiment and initialize model.
        exp = simple_supervised_config["experiment_class"]()
        exp.setup_experiment(simple_supervised_config)

        # Loop through some pseudo epochs.
        for _ in range(5):
            ret = exp.run_epoch()
            self.assertTrue("mean_selected/dendritic_gate" in ret)


if __name__ == "__main__":
    unittest.main(verbosity=2)
