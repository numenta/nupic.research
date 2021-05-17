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
Base Experiment configuration.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import (
    ContextDependentPermutedMNIST,
    PermutedMNIST,
)
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins


class PermutedMNISTExperiment(mixins.RezeroWeights,
                              mixins.PermutedMNISTTaskIndices,
                              DendriteContinualLearningExperiment):
    pass


class CentroidExperiment(mixins.RezeroWeights,
                         mixins.CentroidContext,
                         mixins.PermutedMNISTTaskIndices,
                         ContinualLearningExperiment):
    pass


NUM_TASKS = 10

# A relatively quick running experiment for debugging
DEFAULT_BASE = dict(
    experiment_class=PermutedMNISTExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=ContextDependentPermutedMNIST,
    dataset_args=dict(
        num_tasks=NUM_TASKS,
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        dim_context=1024,
        seed=42,
        download=True,  # Change to True if running for the first time
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[64, 64],
        num_segments=NUM_TASKS,
        dim_context=1024,  # Note: with the Gaussian dataset, `dim_context` was
        # 2048, but this shouldn't effect results
        kw=True,
        # dendrite_sparsity=0.0,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=1,
    tasks_to_validate=(0, 1, 2),  # Tasks on which to run validate
    epochs_to_validate=[],
    num_tasks=NUM_TASKS,
    num_classes=10 * NUM_TASKS,
    distributed=False,
    seed=42,

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=0.001),
)

# Temporary, just for testing
BASE2 = deepcopy(DEFAULT_BASE)
BASE2.update(
    epochs=tune.grid_search([1, 2, 3]),
)

# 10 tasks. Currently several variables depend on tasks and are hardcoded in config
BASE_10_TASKS = deepcopy(DEFAULT_BASE)
BASE_10_TASKS.update(
    dataset_args=dict(
        num_tasks=10,        # NUM_TASKS
        root=os.path.expanduser("~/nta/results/data/"),
        dim_context=1024,
        seed=42,
        download=False,  # Change to True if running for the first time
    ),

    epochs=tune.sample_from(lambda spec: np.random.randint(1, 4)),
    tasks_to_validate=range(100),  # Tasks on which to run validate

    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2048, 2048],
        num_segments=10,    # NUM_TASKS
        dim_context=1024,
        kw=True,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.sample_from(
            lambda spec: np.random.choice([0.95, 0.90, 0.8])),
    ),

    num_tasks=10,            # NUM_TASKS
    num_classes=100,         # 10 * NUM_TASKS
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=20,

    optimizer_class=tune.grid_search([torch.optim.Adam, torch.optim.SGD]),
    optimizer_args=dict(
        lr=tune.sample_from(
            lambda spec: np.random.choice([0.01, 0.005, 0.001, 0.0005])),
    ),
)

# 10 tasks. Currently several variables depend on tasks and are hardcoded in config
BASE_10_SPARSITY_SEARCH = deepcopy(BASE_10_TASKS)
BASE_10_SPARSITY_SEARCH.update(
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2048, 2048],
        num_segments=10,    # NUM_TASKS
        dim_context=1024,
        kw=True,
        kw_percent_on=tune.sample_from(
            lambda spec: np.random.choice([0.05, 0.1, 0.2, 0.0])),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.sample_from(
            lambda spec: np.random.choice([0.90, 0.8, 0.7, 0.5, 0.25, 0.0])),
    ),

    epochs=tune.sample_from(lambda spec: np.random.randint(1, 3)),
    num_samples=5,

    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(
        lr=tune.sample_from(lambda spec: np.random.choice([0.001, 0.0005])),
    ),
)

# 50 tasks. Currently several variables depend on tasks and are hardcoded in config
BASE_50_SPARSITY_SEARCH = deepcopy(BASE_10_SPARSITY_SEARCH)
BASE_50_SPARSITY_SEARCH.update(
    dataset_args=dict(
        num_tasks=50,        # NUM_TASKS
        root=os.path.expanduser("~/nta/results/data/"),
        dim_context=1024,
        seed=42,
        download=False,  # Change to True if running for the first time
    ),

    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2048, 2048],
        num_segments=50,    # NUM_TASKS
        dim_context=1024,
        kw=True,
        kw_percent_on=tune.sample_from(
            lambda spec: np.random.choice([0.05, 0.1, 0.2, 0.0])),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.sample_from(
            lambda spec: np.random.choice([0.90, 0.8, 0.5, 0.0])),
    ),

    tasks_to_validate=(0, 1, 40, 49, 50),  # Tasks on which to run validate
    epochs=1,
    num_samples=8,
    num_tasks=50,  # NUM_TASKS
    num_classes=500,  # 10 * NUM_TASKS

    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(
        lr=tune.sample_from(lambda spec: np.random.choice([0.001, 0.0005])),
    ),
)

# Centroid method for inferring contexts: 10 permutedMNIST tasks
CENTROID_10 = dict(
    experiment_class=CentroidExperiment,
    num_samples=8,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
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
        kw_percent_on=0.1,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=3,
    tasks_to_validate=(0, 1, 4, 9, 24, 49),
    num_tasks=10,
    num_classes=10 * 10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=5e-4),
)

# Centroid method for inferring contexts: 50 permutedMNIST tasks
CENTROID_50 = deepcopy(CENTROID_10)
CENTROID_50["dataset_args"].update(num_tasks=50)
CENTROID_50["model_args"].update(num_segments=50)
CENTROID_50.update(
    epochs=2,
    num_tasks=50,
    num_classes=10 * 50,
)


# Export configurations in this file
CONFIGS = dict(
    default_base=DEFAULT_BASE,
    base_10_tasks=BASE_10_TASKS,
    base_10_sparsity=BASE_10_SPARSITY_SEARCH,
    base_50_search=BASE_50_SPARSITY_SEARCH,
    base2=BASE2,
    centroid_10=CENTROID_10,
    centroid_50=CENTROID_50
)
