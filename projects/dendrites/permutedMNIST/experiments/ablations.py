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
Experiments designed to investigate different dendritic functions that mix feedforward
and dendritic inputs. Examples include additive bias, multiplicative, multiplicative
gating, etc.
"""

from copy import deepcopy
import os
import ray.tune as tune
import torch
import torch.nn.functional as F
import numpy as np

from nupic.research.frameworks.dendrites import (
    DendriticMLP,
)

from nupic.research.frameworks.dendrites.dendrite_cl_experiment import (
    DendriteContinualLearningExperiment,
)

from nupic.research.frameworks.dendrites.modules.dendritic_layers import (
    ZeroSegmentDendriticLayer,
)

from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import mixins
from nupic.research.frameworks.dendrites.mixins import EvalPerTask


from .no_dendrites import NoDendriteExperiment, SPARSE_CL_2

class NoDendriteExperimentEvalPerTask(EvalPerTask, NoDendriteExperiment):
    pass


DENSE_CL_10_NO_DENDRITES = deepcopy(SPARSE_CL_2)
DENSE_CL_10_NO_DENDRITES.update(
    num_samples=1,
    dataset_args=dict(
        num_tasks=10,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        seed=42,
    ),
    model_args=dict(
        input_size=784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
    batch_size=128,
    val_batch_size=512,
    epochs=1,
    tasks_to_validate=[0, 1, 2, 3, 4, 9, 24, 49, 74, 99],
    num_classes=10 * 10,  # TODO: do we need to specify this, num_tasks under dataset args as well as out here?
    num_tasks=10,
    distributed=False,
    # code broke with this line, so I commented it out. But this might mean
    # random seed is always 42.
    # seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                       # SGD with default hyperparameter settings
    optimizer_args=dict(lr=tune.grid_search([5e-4, 5e-3, 5e-2])),

    # For wandb
    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="DENSE_CL_10_NO_DENDRITES",
            group="DENDRITE_ABLATIONS",
        ),
    ),
)

# Note that this is using static sparsity where the non-0 params are decided
# once in advance, so we should maybe average over multiple runs to account
# for variation in the static masks. I'm not gonna do that for now to save
# some compute.
SPARSE_CL_10_NO_DENDRITES = deepcopy(DENSE_CL_10_NO_DENDRITES)
SPARSE_CL_10_NO_DENDRITES["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
)
SPARSE_CL_10_NO_DENDRITES["env_config"]["wandb"].update(
    name="SPARSE_CL_10_NO_DENDRITES",
)

DENSE_CL_10_KW_NO_DENDRITES = deepcopy(DENSE_CL_10_NO_DENDRITES)
DENSE_CL_10_KW_NO_DENDRITES["model_args"].update(
    kw=True,
    kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
)
DENSE_CL_10_KW_NO_DENDRITES["env_config"]["wandb"].update(
    name="DENSE_CL_10_KW_NO_DENDRITES",
)

SPARSE_CL_10_KW_NO_DENDRITES = deepcopy(SPARSE_CL_10_NO_DENDRITES)
SPARSE_CL_10_KW_NO_DENDRITES.update(
    kw=True,
    kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
)
SPARSE_CL_10_KW_NO_DENDRITES["env_config"]["wandb"].update(
    name="SPARSE_CL_10_KW_NO_DENDRITES",
)

DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES = deepcopy(DENSE_CL_10_NO_DENDRITES)
DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["dataset_args"].update(
    cat_one_hot_context=True
)
DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["env_config"]["wandb"].update(
    name="DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES",
)
DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["model_args"].update(
    input_size=784+10,  # original dimension + 1 for each task
)
DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES.update(
    optimizer_args=dict(lr=5e-3)
)

# TODO: figure out how to concatenate task_id as context
CONFIGS = dict(
    dense_cl_10_no_dendrites=DENSE_CL_10_NO_DENDRITES,
    sparse_cl_10_no_dendrites=SPARSE_CL_10_NO_DENDRITES,
    dense_cl_10_kw_no_dendrites=DENSE_CL_10_KW_NO_DENDRITES,
    sparse_cl_10_kw_no_dendrites=SPARSE_CL_10_KW_NO_DENDRITES,
    dense_cl_10_one_hot_ctx_no_dendrites=DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES,
)