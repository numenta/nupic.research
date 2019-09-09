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

from ray import tune

from dynamic_sparse.common.loggers import DEFAULT_LOGGERS
from dynamic_sparse.common.utils import run_ray

# experiment configurations
base_exp_config = dict(
    device="cuda",
    # dataset related
    dataset_name="MNIST",
    data_dir=os.path.expanduser("~/nta/datasets"),
    input_size=784,
    num_classes=10,
    # network related
    network="MLPHeb",
    hidden_sizes=[100, 100, 100],
    batch_norm=True,
    use_kwinners=False,
    # model related
    model="DSNNMixedHeb",
    on_perc=0.2,
    optim_alg="SGD",
    momentum=0.9,
    weight_decay=1e-4,
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    lr_milestones=[30, 60, 90],
    lr_gamma=0.1,
    # sparse related
    hebbian_prune_perc=None,
    weight_prune_perc=0.3,
    pruning_early_stop=0,
    hebbian_grow=tune.grid_search([True, False]),
    # additional validation
    test_noise=False,
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
tune_config = dict(
    name=__file__.replace(".py", "") + "_test13",
    num_samples=8,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 30},
    resources_per_trial={"cpu": 1, "gpu": 0.25},
    loggers=DEFAULT_LOGGERS,
    verbose=0,
)


run_ray(tune_config, base_exp_config)
