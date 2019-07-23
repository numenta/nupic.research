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

import ray
import ray.tune as tune
import torch

from loggers import DEFAULT_LOGGERS
from utils import Trainable, new_experiment, run_experiment

torch.manual_seed(32)


# experiment configurations
base_exp_config = dict(
    device="cuda",
    # dataset related
    dataset_name="CIFAR10",
    input_size=3072,
    num_classes=10,
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir="~/nta/datasets",
    # model related
    model="DSNNMixedHeb",
    network="MLPHeb",
    init_weights=True,
    batch_norm=True,
    dropout=False,
    kwinners=True,
    percent_on=0.3,
    boost_strength=1.4,
    boost_strength_factor=0.7,
    # optimizer related
    optim_alg="SGD",
    momentum=0.9,
    learning_rate=0.01,
    weight_decay=1e-4,
    # sparse related
    epsilon=100,
    start_sparse=1,
    end_sparse=None,
    weight_prune_perc=0.45,
    hebbian_prune_perc=0.45,
    pruning_es=True,
    pruning_es_patience=0,
    pruning_es_window_size=5,
    pruning_es_threshold=0.02,
    pruning_interval=1,
    # additional validation
    test_noise=True,
    noise_level=0.1,
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
tune_config = dict(
    name="hebbian-gs-test",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 1000},  # 300 in cifar
    resources_per_trial={"cpu": 1, "gpu": 0.33},
    loggers=DEFAULT_LOGGERS,
    verbose=1,
)

# define experiments
experiments = {
    "baselines": dict(
        model=tune.grid_search(["BaseModel", "SparseModel", "DSNN"]),
        weight_prune_perc=0.3,
    ),  # 3
    "mixed_hebbian_gs": dict(
        weight_prune_perc=tune.grid_search([0.15, 0.30, 0.45, 0.60]),
        hebbian_prune_perc=tune.grid_search([0.15, 0.30, 0.45, 0.60]),
    ),  # 16
    "epsilons": dict(epsilon=tune.grid_search([30, 200])),  # 2
    "spaced_updates": dict(pruning_interval=tune.grid_search([2, 4, 8, 16, 32])),  # 5
    "es_strategies": dict(pruning_es_patience=tune.grid_search([2, 1000])),  # 2
}
exp_configs = [
    (name, new_experiment(base_exp_config, c)) for name, c in experiments.items()
]

# run all experiments in parallel
ray.init()
results = [
    run_experiment.remote(name, Trainable, c, tune_config) for name, c in exp_configs
]
ray.get(results)
ray.shutdown()
