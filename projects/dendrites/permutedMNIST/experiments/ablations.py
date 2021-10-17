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

from nupic.research.frameworks.dendrites.mixins import EvalPerTask
from nupic.research.frameworks.pytorch.datasets import OneHotContextPermutedMNIST, PermutedMNIST
from nupic.research.frameworks.pytorch.models import ModifiedInitStandardMLP


from .mlp import MLPExperiment, THREE_LAYER_MLP_10

class MLPExperimentEvalPerTask(EvalPerTask, MLPExperiment):
    pass


THREE_LAYER_MLP_10_ONEHOT = deepcopy(THREE_LAYER_MLP_10)
THREE_LAYER_MLP_10_ONEHOT.update(
    dataset_class=OneHotContextPermutedMNIST,
    num_samples=1,
    num_tasks=10,
    num_classes=10 * 10,
    model_class=ModifiedInitStandardMLP,
    model_args=dict(
        input_size=784 + 10,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        num_classes=10 * 10,
    ),

    optimizer_args=dict(lr=tune.grid_search([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])),
    # For wandb
    env_config=dict(
        wandb=dict(
            entity="nupic-research",
            project="dendrite_baselines",
            name="THREE_LAYER_MLP_10_ONEHOT",
            group="MLP_ABLATIONS",
        ),
    ),
)

# # Note that this is using static sparsity where the non-0 params are decided
# # once in advance, so we should maybe average over multiple runs to account
# # for variation in the static masks. I'm not gonna do that for now to save
# # some compute.
THREE_LAYER_MLP_10_ONEHOT_SPARSE = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_ONEHOT_SPARSE["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, .2, 0.5, .75, 0.9]),  # total hack: wandb is erroring out exactly on every
    # other trial, seems like it has to do with wandb.init and wandb.shutdown, so to get the info I want, just
    # add bogus trials every other (.2, 0.75)
)
THREE_LAYER_MLP_10_ONEHOT_SPARSE["env_config"]["wandb"].update(
    name="THREE_LAYER_MLP_10_ONEHOT_SPARSE",
)
THREE_LAYER_MLP_10_ONEHOT_SPARSE.update(
    optimizer_args=dict(lr=3e-6)
)

# DENSE_CL_10_KW_NO_DENDRITES = deepcopy(DENSE_CL_10_NO_DENDRITES)
# DENSE_CL_10_KW_NO_DENDRITES["model_args"].update(
#     kw=True,
#     kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
# )
# DENSE_CL_10_KW_NO_DENDRITES["env_config"]["wandb"].update(
#     name="DENSE_CL_10_KW_NO_DENDRITES",
# )

# SPARSE_CL_10_KW_NO_DENDRITES = deepcopy(SPARSE_CL_10_NO_DENDRITES)
# SPARSE_CL_10_KW_NO_DENDRITES.update(
#     kw=True,
#     kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
# )
# SPARSE_CL_10_KW_NO_DENDRITES["env_config"]["wandb"].update(
#     name="SPARSE_CL_10_KW_NO_DENDRITES",
# )

# DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES = deepcopy(DENSE_CL_10_NO_DENDRITES)
# # DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["dataset_args"].update(
# #     cat_one_hot_context=True
# # )
# DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["env_config"]["wandb"].update(
#     name="DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES",
# )
# DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["model_args"].update(
#     input_size=784+10,  # original dimension + 1 for each task
#     hidden_sizes=[2048, 2048],  # hidden layers get context too
#     weight_sparsity=0.,
#     kw=False,
# )
# DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES.update(
#     optimizer_args=dict(lr=5e-4),
#     dataset_class=OneHotContextPermutedMNIST,
#     # experiment_class=NoDendriteExperimentOneHotContextEvalPerTask,
# )

# SPARSE_CL_10_ONE_HOT_CTX_NO_DENDRITES= deepcopy(DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES)
# SPARSE_CL_10_ONE_HOT_CTX_NO_DENDRITES["model_args"].update(
#     weight_sparsity=tune.grid_search([0.1, 0.5, 0.9])
# )

# DENSE_CL_10_KW_ONE_HOT_CTX_NO_DENDRITES = deepcopy(DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES)
# DENSE_CL_10_KW_ONE_HOT_CTX_NO_DENDRITES["model_args"].update(
#     kw=True,
#     kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5])
# )

# SPARSE_CL_10_KW_ONE_HOT_CTX_NO_DENDRITES = deepcopy(SPARSE_CL_10_ONE_HOT_CTX_NO_DENDRITES)
# SPARSE_CL_10_KW_ONE_HOT_CTX_NO_DENDRITES["model_args"].update(
#     weight_sparsity=0.9,
#     kw=True,
#     kw_percent_on=tune.grid_search([.01, .1, .3])
# )


# DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES_DEBUG = deepcopy(DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES)
# DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES_DEBUG.update(
#     experiment_class=NoDendriteExperimentEvalPerTask,
#     # num_tasks is both a dataset arg and an arg to cl_experiment
#     num_tasks=2,
#     num_classes=10*2,
#     dataset_args=dict(
#         num_tasks=2,
#         root=os.path.expanduser("~/nta/results/data/"),
#         download=True,  # Change to True if running for the first time
#         seed=42,
#     ),
#     model_args=dict(
#         input_size=784 + 2,
#         output_size=10,  # Single output head shared by all tasks
#         hidden_sizes=[2048, 2048],
#         num_segments=0,
#         dim_context=0,
#         kw=False,
#         dendrite_weight_sparsity=0.0,
#         weight_sparsity=0.,
#         context_percent_on=0.0,
#         dendritic_layer_class=ZeroSegmentDendriticLayer,
#     )
# )

# TODO: figure out how to concatenate task_id as context
CONFIGS = dict(
    three_layer_mlp_10_onehot=THREE_LAYER_MLP_10_ONEHOT,
    # three_layer_mlp_10_onehot_sparse=THREE_LAYER_MLP_10_ONEHOT_SPARSE,
    # sparse_cl_10_no_dendrites=SPARSE_CL_10_NO_DENDRITES,
    # dense_cl_10_kw_no_dendrites=DENSE_CL_10_KW_NO_DENDRITES,
    # sparse_cl_10_kw_no_dendrites=SPARSE_CL_10_KW_NO_DENDRITES,
    # dense_cl_10_one_hot_ctx_no_dendrites=DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES,
    # dense_cl_10_one_hot_ctx_no_dendrites_debug=DENSE_CL_10_ONE_HOT_CTX_NO_DENDRITES_DEBUG,
    # dense_cl_10_kw_one_hot_ctx_no_dendrites=DENSE_CL_10_KW_ONE_HOT_CTX_NO_DENDRITES,
    # sparse_cl_10_one_hot_ctx_no_dendrites=SPARSE_CL_10_ONE_HOT_CTX_NO_DENDRITES,
    # sparse_cl_10_kw_one_hot_ctx_no_dendrites=SPARSE_CL_10_KW_ONE_HOT_CTX_NO_DENDRITES
)