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

"""
Continual Learning experiments with standard MLPs.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.pytorch.models import StandardMLP
from nupic.research.frameworks.vernon import mixins
from nupic.torch.functions import kwinners


def kwinners_wrapper(y):
    return kwinners(y, duty_cycles=None, k=102, boost_strength=0)


class MLPExperiment(mixins.PermutedMNISTTaskIndices,
                    DendriteContinualLearningExperiment):
    pass


class ContinualLearningMLP(StandardMLP):
    """
    An MLP used for continual learning experiments on permutedMNIST. The weight
    initialization has been modified to improve performance.
    """

    def __init__(self, input_size, num_classes, hidden_sizes):
        super().__init__(input_size, num_classes, hidden_sizes)

        # Modified Kaiming weight initialization which considers 1) the density of
        # the input vector and 2) the weight density in addition to the fan-in

        weight_density = 1.0
        input_flag = False
        for layer in self.classifier:
            if isinstance(layer, torch.nn.Linear):

                # Assume input is fully dense, but hidden layer activations are only
                # 50% dense due to ReLU
                input_density = 1.0 if not input_flag else 0.5
                input_flag = True

                _, fan_in = layer.weight.size()
                bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
                torch.nn.init.uniform_(layer.weight, -bound, bound)


DEFAULT = dict(
    experiment_class=MLPExperiment,
    local_dir=os.path.expanduser("~/nta/results/experiments/mlp"),
    num_samples=8,

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        root=os.path.expanduser("~/nta/results/data/"),
        seed=42,
        download=False,
    ),

    model_args=dict(input_size=784, num_classes=10),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[9, 99],

    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
)


# MLP with 3 layers on 10 permutedMNIST tasks
THREE_LAYER_MLP_10 = deepcopy(DEFAULT)
THREE_LAYER_MLP_10["dataset_args"].update(num_tasks=10)
THREE_LAYER_MLP_10["model_args"].update(hidden_sizes=[2048, 2048])
THREE_LAYER_MLP_10.update(
    model_class=ContinualLearningMLP,

    num_tasks=10,
    num_classes=10 * 10,

    epochs=5,
    optimizer_args=dict(lr=3e-6)
)


# MLP with 3 layers on 100 permutedMNIST tasks
THREE_LAYER_MLP_100 = deepcopy(THREE_LAYER_MLP_10)
THREE_LAYER_MLP_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    epochs=3,
    optimizer_args=dict(lr=1e-6)
)


# MLP with 10 layers on 10 permutedMNIST tasks
TEN_LAYER_MLP_10 = deepcopy(DEFAULT)
TEN_LAYER_MLP_10["dataset_args"].update(num_tasks=100)
TEN_LAYER_MLP_10["model_args"].update(
    hidden_sizes=[2048 for _ in range(9)]
)
TEN_LAYER_MLP_10.update(
    model_class=ContinualLearningMLP,

    num_tasks=10,
    num_classes=10 * 10,

    epochs=3,
    optimizer_args=dict(lr=3e-6)
)


# MLP with 10 layers on 100 permutedMNIST tasks
TEN_LAYER_MLP_100 = deepcopy(TEN_LAYER_MLP_10)
TEN_LAYER_MLP_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    epochs=3,
    optimizer_args=dict(lr=3e-7)
)


CONFIGS = dict(
    three_layer_mlp_10=THREE_LAYER_MLP_10,
    three_layer_mlp_100=THREE_LAYER_MLP_100,
    ten_layer_mlp_10=TEN_LAYER_MLP_10,
    ten_layer_mlp_100=TEN_LAYER_MLP_100
)
