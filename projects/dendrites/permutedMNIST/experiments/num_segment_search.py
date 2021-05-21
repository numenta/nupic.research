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
Search hyperparameter configuration
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
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST, PermutedMNIST
from nupic.research.frameworks.vernon import mixins, ContinualLearningExperiment

"""Permuted MNIST with DendriticMLP"""


class NbSegmentSearchExperiment(mixins.RezeroWeights,
                                mixins.CentroidContext,
                                mixins.PermutedMNISTTaskIndices,
                                DendriteContinualLearningExperiment):
    pass

NUM_TASKS = 10

NB_SEGMENT_SEARCH = dict(
    experiment_class=NbSegmentSearchExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites/weights_sparsity_search2"),

    # dataset_class=ContextDependentPermutedMNIST,
    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=NUM_TASKS,
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        seed=42,
        download=False,  # Change to True if running for the first time
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2048, 2048],
        num_segments=tune.grid_search([2, 3, 5, 7, 10]),
        dim_context=784,
        kw=True,
        kw_percent_on = tune.grid_search([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
        dendrite_weight_sparsity=0.0,
        weight_sparsity = tune.grid_search([0.1, 0.5, 0.7, 0.9, 0.95, 0.99]),
        context_percent_on=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=3,
    tasks_to_validate= range(NUM_TASKS),  # Tasks on which to run validate
    epochs_to_validate=[],
    num_tasks=NUM_TASKS,
    num_classes=10 * NUM_TASKS,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples = 10,

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=5e-4),
)

# varying only num segments
NB_SEGMENT_SEARCH_2 = deepcopy(NB_SEGMENT_SEARCH)
NB_SEGMENT_SEARCH_2['model_args'].update(
    kw_percent_on = 0.1,
    weight_sparsity = 0.5
)

#need to add a bit more data points
NB_SEGMENT_SEARCH_3 = deepcopy(NB_SEGMENT_SEARCH_2)
NB_SEGMENT_SEARCH_3['model_args'].update(
    num_segments = tune.grid_search([14, 20])
)

# varying only kw sparsity
KW_SPARSITY_SEARCH = deepcopy(NB_SEGMENT_SEARCH)
KW_SPARSITY_SEARCH['model_args'].update(
    num_segments = 10,
    weight_sparsity = 0.5
)

# varying only weights sparsity
W_SPARSITY_SEARCH = deepcopy(NB_SEGMENT_SEARCH)
W_SPARSITY_SEARCH['model_args'].update(
    num_segments = 10,
    kw_percent_on = 0.1
)

# adding a couple of parameter
W_SPARSITY_SEARCH2 = deepcopy(W_SPARSITY_SEARCH)
W_SPARSITY_SEARCH2['model_args'].update(
    weight_sparsity = tune.grid_search([0.05, 0.01])
)

# Export configurations in this file
CONFIGS = dict(
    nb_segment_search = NB_SEGMENT_SEARCH,
    nb_segment_search2 = NB_SEGMENT_SEARCH_2,
    nb_segment_search3 = NB_SEGMENT_SEARCH_3,
    kw_sparsity_search = KW_SPARSITY_SEARCH, #old name is sparsity_search
    weights_sparsity_search = W_SPARSITY_SEARCH,
    weights_sparsity_search2 = W_SPARSITY_SEARCH2
)
