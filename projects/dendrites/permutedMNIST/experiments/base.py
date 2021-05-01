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

import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST
from nupic.research.frameworks.vernon import mixins


class PermutedMNISTExperiment(mixins.RezeroWeights,
                              DendriteContinualLearningExperiment):
    pass


NUM_TASKS = 2

DEFAULT_BASE = dict(
    experiment_class=PermutedMNISTExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=ContextDependentPermutedMNIST,
    dataset_args=dict(
        num_tasks=NUM_TASKS,
        root="~/nta/data/dendrites",
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
    batch_size=64,
    epochs=2,
    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(lr=0.001),
)

# Export configurations in this file
CONFIGS = dict(
    default_base=DEFAULT_BASE,
    base2=BASE2,
)
