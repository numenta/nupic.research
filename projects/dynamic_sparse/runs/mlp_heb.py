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

from dynamic_sparse.common import DEFAULT_LOGGERS
from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# experiment configurations
base_exp_config = dict(
    device="cuda",
    # dataset related
    dataset_name="MNIST",
    input_size=784,
    num_classes=10,
    hidden_sizes=[100, 100, 100],
    data_dir=os.path.expanduser("~/nta/datasets"),
    # model related
    model="DSNNMixedHeb",
    network="MLPHeb",
    kwinners=tune.grid_search([True, False]),  # 2
    percent_on=0.3,
    on_perc=tune.grid_search([0.05, 0.1, 0.2]),  # 3
    # sparse related
    hebbian_prune_perc=tune.grid_search([0, 0.1, 0.2, 0.3, 0.4, 0.5]),  # 6
    weight_prune_perc=tune.grid_search([0, 0.1, 0.2, 0.3, 0.4, 0.5]),  # 6
    pruning_es=False,
    pruning_active=True,
    hebbian_grow=tune.grid_search([True, False]),  # 2
    # additional validation
    test_noise=True,
    noise_level=0.15,  # test with more agressive noise
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
tune_config = dict(
    name=os.path.basename(__file__).replace(".py", "") + "_eval2",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 30},
    resources_per_trial={"cpu": 1, "gpu": 1},
    loggers=DEFAULT_LOGGERS,
    verbose=1,
)

run_ray(tune_config, base_exp_config)
