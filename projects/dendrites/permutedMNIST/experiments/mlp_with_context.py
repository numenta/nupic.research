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

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.dendrites.mixins import EvalPerTask
from nupic.research.frameworks.dendrites.modules.dendritic_layers import (
    ZeroSegmentDendriticLayer,
)
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST, PermutedMNIST
from nupic.research.frameworks.pytorch.models import ModifiedInitStandardMLP, SparseMLP


from .mlp import MLPExperiment, THREE_LAYER_MLP_10
from.no_dendrites import NoDendriteExperiment

# class MLPExperimentEvalPerTask(EvalPerTask, MLPExperiment):
#     pass


### Notes on how this file is organized
# Configs alternate onehot and centroid, adding weight and kwinners sparsity as you go.
# I realized I should have used zero segment dendritic layer instead of SparseMLP, but
# I'm keeping the original names for those to avoid confusion. New experiments will be
# called ZeroSegment.

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
        kw_percent_on=(1., 1.),
        weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9),]),
        input_size=784 + 784,  # + 10 due to 10 tasks
        hidden_sizes=[2048, 2048],
        output_size=10 * 10,
    )
)
THREE_LAYER_MLP_10_CENTROID_SPARSE["model_args"].update(
    weight_sparsity=tune.grid_search([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9),]),
)
THREE_LAYER_MLP_10_CENTROID_SPARSE["env_config"]["wandb"].update(
    name="THREE_LAYER_MLP_10_CENTROID_SPARSE",
)

THREE_LAYER_MLP_10_ONEHOT_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT_SPARSE)
THREE_LAYER_MLP_10_ONEHOT_DENSE_KW["model_args"].update(
    weight_sparsity=(0., 0.),
    kw_percent_on=tune.grid_search([(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)])
)

# This was the rotten apple that didn't update model args properly!
THREE_LAYER_MLP_10_CENTROID_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_CENTROID_SPARSE)
THREE_LAYER_MLP_10_CENTROID_DENSE_KW["model_args"].update(
    weight_sparsity=(0., 0.),
    kw_percent_on=tune.grid_search([(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)])
)

THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT_SPARSE)
THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW["model_args"].update(
    kw_percent_on=tune.grid_search([(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)]),
)

THREE_LAYER_MLP_10_CENTROID_SPARSE_KW = deepcopy(THREE_LAYER_MLP_10_CENTROID_SPARSE)
THREE_LAYER_MLP_10_CENTROID_SPARSE_KW["model_args"].update(
    kw_percent_on=tune.grid_search([(.01, 0.1), (.05, .05), (.1, .1), (.25, .25), (.5, .5)]),
)

### Zero segment instead of SparseMLP

THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW = deepcopy(THREE_LAYER_MLP_10_ONEHOT_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW.update(
    experiment_class=NoDendriteExperiment,
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784 + 10,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)

THREE_LAYER_ZERO_SEGMENT_10_CENTROID_DENSE_KW_ = deepcopy(THREE_LAYER_MLP_10_CENTROID_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_CENTROID_DENSE_KW_.update(
    experiment_class=NoDendriteExperiment,
    model_class=DendriticMLP,
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=tune.grid_search([.01, .05, .1, .25, .5]),
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)

THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE = deepcopy(THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE.update(
    model_args=dict(
        input_size=784 + 10,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)

THREE_LAYER_ZERO_SEGMENT_10_CENTROID_SPARSE_ = deepcopy(THREE_LAYER_ZERO_SEGMENT_10_CENTROID_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_CENTROID_SPARSE_.update(
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=False,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=tune.grid_search([0.1, 0.5, 0.9]),
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    )
)


THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_KW = deepcopy(THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW)
THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_KW["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9])
)

THREE_LAYER_ZERO_SEGMENT_10_CENTROID_SPARSE_KW_ = deepcopy(THREE_LAYER_ZERO_SEGMENT_10_CENTROID_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_10_CENTROID_SPARSE_KW_["model_args"].update(
    weight_sparsity=tune.grid_search([0.1, 0.5, 0.9])
)

THREE_LAYER_ZERO_SEGMENT_30_CENTROID_DENSE_KW_ = deepcopy(THREE_LAYER_ZERO_SEGMENT_10_CENTROID_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_30_CENTROID_DENSE_KW_.update(
    dataset_args=dict(
        num_tasks=30,
        download=True,
        seed=np.random.randint(2, 10_000),
        context_type="centroid",
        combine_context_as="concatenate",
    ),
    model_args=dict(
        input_size=784 + 784,
        output_size=10,  # Single output head shared by all tasks
        hidden_sizes=[2048, 2048],
        num_segments=0,
        dim_context=0,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.,
        context_percent_on=0.0,
        dendritic_layer_class=ZeroSegmentDendriticLayer,
    ),
    num_classes=10 * 30,
    optimizer_args=dict(lr=0.0001),
)

THREE_LAYER_ZERO_SEGMENT_50_CENTROID_DENSE_KW_ = deepcopy(THREE_LAYER_ZERO_SEGMENT_30_CENTROID_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_50_CENTROID_DENSE_KW_.update(
    num_classes=10 * 50,
)
THREE_LAYER_ZERO_SEGMENT_50_CENTROID_DENSE_KW_["dataset_args"].update(
    num_tasks=50
)


THREE_LAYER_ZERO_SEGMENT_100_CENTROID_DENSE_KW_ = deepcopy(THREE_LAYER_ZERO_SEGMENT_30_CENTROID_DENSE_KW_)
THREE_LAYER_ZERO_SEGMENT_100_CENTROID_DENSE_KW_.update(
    num_classes=10 * 100,
)
THREE_LAYER_ZERO_SEGMENT_100_CENTROID_DENSE_KW_["dataset_args"].update(
    num_tasks=100
)


CONFIGS = dict(
    # onehot context mlp
    three_layer_mlp_10_onehot=THREE_LAYER_MLP_10_ONEHOT,
    three_layer_mlp_10_onehot_sparse=THREE_LAYER_MLP_10_ONEHOT_SPARSE,
    three_layer_mlp_10_onehot_dense_kw=THREE_LAYER_MLP_10_ONEHOT_DENSE_KW,
    three_layer_mlp_10_onehot_sparse_kw=THREE_LAYER_MLP_10_ONEHOT_SPARSE_KW,

    # centroid context mlp
    three_layer_mlp_10_centroid=THREE_LAYER_MLP_10_CENTROID,
    three_layer_mlp_10_centroid_sparse=THREE_LAYER_MLP_10_CENTROID_SPARSE,
    three_layer_mlp_10_centroid_dense_kw=THREE_LAYER_MLP_10_CENTROID_DENSE_KW,
    three_layer_mlp_10_centroid_sparse_kw=THREE_LAYER_MLP_10_CENTROID_SPARSE_KW,

    # Zero segment onehot context
    three_layer_zero_segment_10_onehot_dense_kw=THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_DENSE_KW,
    three_layer_zero_segment_10_onehot_sparse=THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE,
    three_layer_zero_segment_10_onehot_sparse_kw=THREE_LAYER_ZERO_SEGMENT_10_ONEHOT_SPARSE_KW,

    # Zero segment centroid context
    three_layer_zero_segment_10_centroid_dense_kw_=THREE_LAYER_ZERO_SEGMENT_10_CENTROID_DENSE_KW_,
    three_layer_zero_segment_10_centroid_sparse_=THREE_LAYER_ZERO_SEGMENT_10_CENTROID_SPARSE_,
    three_layer_zero_segment_10_centroid_sparse_kw_=THREE_LAYER_ZERO_SEGMENT_10_CENTROID_SPARSE_KW_,

    # Scan number of tasks
    three_layer_zero_segment_30_centroid_dense_kw_=THREE_LAYER_ZERO_SEGMENT_30_CENTROID_DENSE_KW_,
    three_layer_zero_segment_50_centroid_dense_kw_=THREE_LAYER_ZERO_SEGMENT_50_CENTROID_DENSE_KW_,
    three_layer_zero_segment_100_centroid_dense_kw_=THREE_LAYER_ZERO_SEGMENT_100_CENTROID_DENSE_KW_,
)
