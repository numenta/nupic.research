# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.research.frameworks.vernon import (
    MetaContinualLearningExperiment,
    SupervisedExperiment,
    mixins,
)
from nupic.torch.modules import SparseWeights


class SupervisedRezeroWeights(mixins.RezeroWeights,
                              SupervisedExperiment):
    pass


class MetaCLRezeroWeights(mixins.RezeroWeights,
                          MetaContinualLearningExperiment):
    pass


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        in_features = np.prod(input_shape)
        self.flatten = torch.nn.Flatten()
        self.classifier = SparseWeights(
            torch.nn.Linear(in_features, num_classes, bias=False),
            sparsity=0.5,
        )

    def forward(self, x):
        return self.classifier(self.flatten(x))


def fake_data(size=100, image_size=(1, 4, 4), train=False):
    return FakeData(size=size, image_size=image_size, transform=ToTensor())


supervised_experiment = dict(

    experiment_class=SupervisedRezeroWeights,
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

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
)

metacl_experiment = {**supervised_experiment}
metacl_experiment.update(
    experiment_class=MetaCLRezeroWeights,
    fast_params=[".*"],  # <- all params
)


class RezeroWeightTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_simple_model_is_half_sparse(self):

        # Supervised Experiment:
        #   Validate that the fully rezeroed model has exactly 80 on-params
        exp = supervised_experiment["experiment_class"]
        model = exp.create_model(supervised_experiment, "cpu")
        model.classifier.module.weight.data[:] = 1
        model.classifier.rezero_weights()
        total_params, on_params = count_nonzero_params(model)
        assert on_params == 80

        # MetaCL Experiment:
        #   Validate that the fully rezeroed model has exactly 80 on-params
        exp = metacl_experiment["experiment_class"]
        model = exp.create_model(metacl_experiment, "cpu")
        model.classifier.module.weight.data[:] = 1
        model.classifier.rezero_weights()
        total_params, on_params = count_nonzero_params(model)
        assert on_params == 80

    def test_supervised_experiment_with_rezero_weights(self):

        # Setup experiment and initialize model.
        exp = supervised_experiment["experiment_class"]()
        exp.setup_experiment(supervised_experiment)
        total_params, on_params = count_nonzero_params(exp.model)
        assert total_params == 160
        assert on_params <= 80  # Less than as some may be randomly zero.

        # Loop through some pseudo epochs.
        for _ in range(10):

            total_params, on_params = count_nonzero_params(exp.model)
            assert on_params <= 80

            exp.run_epoch()

            total_params, on_params = count_nonzero_params(exp.model)
            assert on_params <= 80

    def test_metacl_experiment_with_rezero_weights(self):

        # Get experiment class.
        exp = metacl_experiment["experiment_class"]()

        # The classes are sampled randomly, so we need a way to make sure
        # the experiment will only sample from what's been randomly generated.
        metacl_experiment.pop("num_classes")
        dataset = exp.load_dataset(metacl_experiment, train=True)
        fast_and_slow_sampler = exp.create_train_sampler(metacl_experiment, dataset)
        replay_sampler = exp.create_replay_sampler(metacl_experiment, dataset)
        metacl_experiment.update(
            fast_and_slow_classes=list(fast_and_slow_sampler.task_indices.keys()),
            replay_classes=list(replay_sampler.task_indices.keys()),
        )

        # Setup experiment and initialize model.
        exp.setup_experiment(metacl_experiment)
        total_params, on_params = count_nonzero_params(exp.model)
        assert total_params == 160
        assert on_params <= 80  # Less than as some may be randomly zero.

        # Loop through some pseudo epochs.
        for _ in range(10):

            total_params, on_params = count_nonzero_params(exp.model)
            assert on_params <= 80

            exp.run_epoch()

            total_params, on_params = count_nonzero_params(exp.model)
            assert on_params <= 80


if __name__ == "__main__":
    unittest.main(verbosity=2)
