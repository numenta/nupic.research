#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Experiment file that runs dendritic networks which use the raw image as context
"""

import os

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import SelfContextPermutedMNIST
from nupic.research.frameworks.vernon import mixins


class SimpleExperiment(mixins.RezeroWeights,
                       mixins.PermutedMNISTTaskIndices,
                       DendriteContinualLearningExperiment):
    pass


SELF_CONTEXT_10 = dict(
    experiment_class=SimpleExperiment,
    num_samples=1,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=SelfContextPermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        root=os.path.expanduser("~/nta/results/data/"),
        download=True,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=DendriticMLP,  # CentroidDendriticMLP does not affect accuracy..??
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=10,
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=tune.grid_search([1, 2, 4, 8]),
    tasks_to_validate=[0, 1, 2, 3, 4, 9, 24, 49, 74, 99],
    num_tasks=10,
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=tune.grid_search([1e-6, 1e-5, 1e-4, 5e-4])),
)


# Export configurations in this file
CONFIGS = dict(
    self_context_10=SELF_CONTEXT_10,
)
