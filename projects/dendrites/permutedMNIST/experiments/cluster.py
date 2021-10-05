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
Experiment file that runs dendritic networks which infer the context vector via 1)
applying a clustering procedure during training, and 2) inferring the closest prototype
as context during inference.
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
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins


class CentroidExperiment(mixins.RezeroWeights,
                         mixins.CentroidContext,
                         mixins.PermutedMNISTTaskIndices,
                         DendriteContinualLearningExperiment):
    pass


CENTROID_CLUSTER_5 = dict(
    experiment_class=CentroidExperiment,
    num_samples=8,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=5,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        seed=42,
    ),

    model_class=DendriticMLP,  # CentroidDendriticMLP does not affect accuracy..??
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=5,
        dim_context=256,
        kw=True,
        kw_percent_on=0.1,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.05,
    ),

    centroid_context_args=dict(infer_while_training=True),

    batch_size=256,
    val_batch_size=512,
    epochs=5,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    num_tasks=5,
    num_classes=10 * 5,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=1e-3),
)

CENTROID_CLUSTER_2 = deepcopy(CENTROID_CLUSTER_5)
CENTROID_CLUSTER_2["dataset_args"].update(num_tasks=2)
CENTROID_CLUSTER_2["model_args"].update(num_segments=2)
CENTROID_CLUSTER_2.update(
    num_tasks=2,
    num_classes=10 * 2,

    optimizer_args=dict(lr=1e-3),
    epochs=5
)

CENTROID_CLUSTER_10 = deepcopy(CENTROID_CLUSTER_5)
CENTROID_CLUSTER_10["dataset_args"].update(num_tasks=10)
CENTROID_CLUSTER_10["model_args"].update(num_segments=10)
CENTROID_CLUSTER_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    optimizer_args=dict(lr=1e-3),
    epochs=3
)

CENTROID_CLUSTER_25 = deepcopy(CENTROID_CLUSTER_5)
CENTROID_CLUSTER_25["dataset_args"].update(num_tasks=25)
CENTROID_CLUSTER_25["model_args"].update(num_segments=25)
CENTROID_CLUSTER_25.update(
    num_tasks=25,
    num_classes=10 * 25,

    optimizer_args=dict(lr=3e-4),
    epochs=1
)

CENTROID_CLUSTER_50 = deepcopy(CENTROID_CLUSTER_5)
CENTROID_CLUSTER_50["dataset_args"].update(num_tasks=50)
CENTROID_CLUSTER_50["model_args"].update(kw_percent_on=0.05, num_segments=50)
CENTROID_CLUSTER_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    optimizer_args=dict(lr=1e-4),
    epochs=3
)

CENTROID_CLUSTER_100 = deepcopy(CENTROID_CLUSTER_5)
CENTROID_CLUSTER_100["dataset_args"].update(num_tasks=100)
CENTROID_CLUSTER_100["model_args"].update(kw_percent_on=0.05, num_segments=100)
CENTROID_CLUSTER_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    optimizer_args=dict(lr=3e-5),
    epochs=5,
)

# Export configurations in this file
CONFIGS = dict(
    centroid_cluster_2=CENTROID_CLUSTER_2,
    centroid_cluster_5=CENTROID_CLUSTER_5,
    centroid_cluster_10=CENTROID_CLUSTER_10,
    centroid_cluster_25=CENTROID_CLUSTER_25,
    centroid_cluster_50=CENTROID_CLUSTER_50,
    centroid_cluster_100=CENTROID_CLUSTER_100,
)
