# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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
Experiment file that run Active Dendrites Network which infer the context vector via
1) applying a clustering procedure during training, and 2) inferring the closest
prototype as context during inference.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F

from nupic.research.frameworks.continual_learning import mixins as cl_mixins
from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites import mixins as dendrites_mixins
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins as vernon_mixins


class PrototypeClusterExperiment(vernon_mixins.RezeroWeights,
                                 dendrites_mixins.PrototypeContext,
                                 cl_mixins.PermutedMNISTTaskIndices,
                                 DendriteContinualLearningExperiment):
    pass


BASE = dict(
    experiment_class=PrototypeClusterExperiment,
    num_samples=1,

    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,
        seed=42,
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2048, 2048],
        dim_context=256,  # For clustering/constructing prototypes, this value is
                          # smaller than when prototypes are based on task labels
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),

    prototype_context_args=dict(construct=True),

    batch_size=256,
    val_batch_size=512,
    tasks_to_validate=[1, 4, 9, 24, 49, 99],
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
)


CONSTRUCT_PROTOTYPE_2 = deepcopy(BASE)
CONSTRUCT_PROTOTYPE_2["dataset_args"].update(num_tasks=2)
CONSTRUCT_PROTOTYPE_2["model_args"].update(num_segments=2)
CONSTRUCT_PROTOTYPE_2.update(
    num_tasks=2,
    num_classes=10 * 2,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=1e-3),
)


CONSTRUCT_PROTOTYPE_5 = deepcopy(BASE)
CONSTRUCT_PROTOTYPE_5["dataset_args"].update(num_tasks=5)
CONSTRUCT_PROTOTYPE_5["model_args"].update(num_segments=5)
CONSTRUCT_PROTOTYPE_5.update(
    num_tasks=5,
    num_classes=10 * 5,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=5,
    optimizer_args=dict(lr=1e-3),
)


CONSTRUCT_PROTOTYPE_10 = deepcopy(BASE)
CONSTRUCT_PROTOTYPE_10["dataset_args"].update(num_tasks=10)
CONSTRUCT_PROTOTYPE_10["model_args"].update(num_segments=10)
CONSTRUCT_PROTOTYPE_10.update(
    num_tasks=10,
    num_classes=10 * 10,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=1e-3),
)


CONSTRUCT_PROTOTYPE_25 = deepcopy(BASE)
CONSTRUCT_PROTOTYPE_25["dataset_args"].update(num_tasks=25)
CONSTRUCT_PROTOTYPE_25["model_args"].update(num_segments=25)
CONSTRUCT_PROTOTYPE_25.update(
    num_tasks=25,
    num_classes=10 * 25,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=1,
    optimizer_args=dict(lr=3e-4),
)


CONSTRUCT_PROTOTYPE_50 = deepcopy(BASE)
CONSTRUCT_PROTOTYPE_50["dataset_args"].update(num_tasks=50)
CONSTRUCT_PROTOTYPE_50["model_args"].update(num_segments=50)
CONSTRUCT_PROTOTYPE_50.update(
    num_tasks=50,
    num_classes=10 * 50,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=1e-4),
)


CONSTRUCT_PROTOTYPE_100 = deepcopy(BASE)
CONSTRUCT_PROTOTYPE_100["dataset_args"].update(num_tasks=100)
CONSTRUCT_PROTOTYPE_100["model_args"].update(num_segments=100)
CONSTRUCT_PROTOTYPE_100.update(
    num_tasks=100,
    num_classes=10 * 100,

    # The following number of training epochs and learning rate were chosen based on a
    # hyperparameter search that maximized final test accuracy across all tasks
    epochs=3,
    optimizer_args=dict(lr=1e-4),
)


# Export configurations in this file
CONFIGS = dict(
    construct_prototype_2=CONSTRUCT_PROTOTYPE_2,
    construct_prototype_5=CONSTRUCT_PROTOTYPE_5,
    construct_prototype_10=CONSTRUCT_PROTOTYPE_10,
    construct_prototype_25=CONSTRUCT_PROTOTYPE_25,
    construct_prototype_50=CONSTRUCT_PROTOTYPE_50,
    construct_prototype_100=CONSTRUCT_PROTOTYPE_100,
)
