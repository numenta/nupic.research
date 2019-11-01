# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import os

import numpy as np
from ray import tune

from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# alternative initialization based on configuration
exp_config = dict(
    device="cuda",
    network="resnet18",
    dataset_name="CIFAR10",
    input_size=(3, 32, 32),
    augment_images=True,
    num_classes=10,
    model="SET",
    # specific pruning
    target_final_density=0.1,
    start_pruning_epoch=1,
    end_pruning_epoch=120,
    epochs=200,
    # sparsity related
    on_perc=tune.grid_search(list(np.arange(0.1, 1.01, 0.05))),
    sparse_start=1,
    sparse_end=None,
    # ---- optimizer related
    optim_alg="SGD",
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    lr_milestones=[60, 120, 160],
    lr_gamma=0.2,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov_momentum=True,
    # ---- optimizer related
    hebbian_prune_perc=None,
    weight_prune_perc=0.25,
    pruning_early_stop=2,
    hebbian_grow=False,
)

# run
tune_config = dict(
    name="comparison_set",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    resources_per_trial={"cpu": 1, "gpu": 0.195},
    verbose=2,
)

run_ray(tune_config, exp_config, fix_seed=True)


# alternative initialization based on configuration
exp_config = dict(
    device="cuda",
    network="resnet18",
    dataset_name="CIFAR10",
    input_size=(3, 32, 32),
    augment_images=True,
    num_classes=10,
    model="PruningModel",
    # specific pruning
    on_perc=1.0,
    start_pruning_epoch=1,
    end_pruning_epoch=120,
    epochs=200,
    # sparsity related
    target_final_density=tune.grid_search(list(np.arange(0.1, 1.01, 0.05))),
    sparse_start=1,
    sparse_end=None,
    # ---- optimizer related
    optim_alg="SGD",
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    lr_milestones=[60, 120, 160],
    lr_gamma=0.2,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov_momentum=True,
)

# run
tune_config = dict(
    name="comparison_pruning",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    resources_per_trial={"cpu": 1, "gpu": 0.195},
    verbose=2,
)

run_ray(tune_config, exp_config, fix_seed=True)
