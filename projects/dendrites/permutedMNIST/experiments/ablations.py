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
import numpy as np
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites.mixins import EvalPerTask
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST, PermutedMNIST
from nupic.research.frameworks.pytorch.models import ModifiedInitStandardMLP, SparseMLP


from .mlp import MLPExperiment, THREE_LAYER_MLP_10

class MLPExperimentEvalPerTask(EvalPerTask, MLPExperiment):
    pass


THREE_LAYER_MLP_10_ONEHOT = deepcopy(THREE_LAYER_MLP_10)
THREE_LAYER_MLP_10_ONEHOT.update(
    dataset_class=ContextDependentPermutedMNIST,
    dataset_args=dict(
        num_tasks=10,
        download=True,
        seed=np.random.randint(2, 10_000),
        context_type="one_hot",
        combine_context_as="concatenate",
    ),
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

THREE_LAYER_MLP_10_CENTROID = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_CENTROID["dataset_args"].update(
    context_type="centroid",
    combine_context_as="concatenate",
)
THREE_LAYER_MLP_10_CENTROID["model_args"].update(
    input_size=784 + 784,  # 784 image + 784 context
)


THREE_LAYER_MLP_10_ONEHOT_SPARSE = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_ONEHOT_SPARSE.update(
    model_class=SparseMLP,
    model_args=dict(
        kw_percent_on=(1., 1.),
        weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9),]),
        input_size=784 + 10,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        output_size=10 * 10,
    )
)

THREE_LAYER_MLP_10_CENTROID_SPARSE = deepcopy(THREE_LAYER_MLP_10_CENTROID)
THREE_LAYER_MLP_10_CENTROID_SPARSE.update(
    model_class=SparseMLP,
    model_args=dict(
        weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9),]),
        input_size=784 + 784,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        output_size=10 * 10,
    )
)
THREE_LAYER_MLP_10_CENTROID_SPARSE["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
)
THREE_LAYER_MLP_10_CENTROID_SPARSE["env_config"]["wandb"].update(
    name="THREE_LAYER_MLP_10_CENTROID_SPARSE",
)

THREE_LAYER_MLP_10_ONEHOT_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT)
THREE_LAYER_MLP_10_ONEHOT_DENSE_KW["model_args"].update(
    weight_sparsity=0.,
    kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5])
)

THREE_LAYER_MLP_10_CENTROID_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_CENTROID_SPARSE)
THREE_LAYER_MLP_10_CENTROID_DENSE_KW.update(
    weight_sparsity=0.,
    kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5])
)

THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT_SPARSE)
THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW["model_args"].update(
    kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
)

THREE_LAYER_MLP_10_CENTROID_SPARSE_KW = deepcopy(THREE_LAYER_MLP_10_CENTROID_DENSE_KW)
THREE_LAYER_MLP_10_CENTROID_SPARSE_KW.update(
    kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
)

CONFIGS = dict(
    three_layer_mlp_10_onehot=THREE_LAYER_MLP_10_ONEHOT,  # done
    three_layer_mlp_10_onehot_sparse=THREE_LAYER_MLP_10_ONEHOT_SPARSE,
    three_layer_mlp_10_onehot_dense_kw=THREE_LAYER_MLP_10_ONEHOT_DENSE_KW,
    three_layer_mlp_10_onehot_sparse_kw=THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW,
    three_layer_mlp_10_centroid=THREE_LAYER_MLP_10_CENTROID,
    three_layer_mlp_10_centroid_sparse=THREE_LAYER_MLP_10_CENTROID_SPARSE,
    three_layer_mlp_10_centroid_dense_kw=THREE_LAYER_MLP_10_CENTROID_DENSE_KW,
    three_layer_mlp_10_centroid_sparse_kw=THREE_LAYER_MLP_10_CENTROID_SPARSE_KW,
)