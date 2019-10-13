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

from nupic.research.frameworks.dynamic_sparse.common.loggers import DEFAULT_LOGGERS
from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# define a small convolutional network
# in line with the

# experiment configurations
base_exp_config = dict(
    device="cuda",
    # dataset related
    dataset_name="PreprocessedGSC",
    data_dir="~/nta/datasets/gsc",
    batch_size_train=(4, 16),
    batch_size_test=1000,
    # network related
    network=tune.grid_search(["GSCHeb_v0", "GSCHeb"]),
    optim_alg="SGD",
    momentum=0,  # 0.9,
    learning_rate=0.01,  # 0.1,
    weight_decay=0.01,  # 1e-4,
    lr_scheduler="MultiStepLR",
    lr_milestones=[30, 60, 90],
    lr_gamma=0.9,  # 0.1,
    use_kwinners=True,
    # sparse_linear_only=True, # False
    # model related
    model="DSNNWeightedMag",
    # on_perc=0.04,
    # sparse related
    on_perc=tune.grid_search([0.02, 0.03, 0.04]),
    weight_prune_perc=0.3,
    # weight_prune_perc=tune.grid_search(list(np.arange(0, 1.001, 0.05))),
    # pruning_early_stop=2,
    # additional validation
    # test_noise=False,
    # debugging
    # debug_weights=True,
    # debug_sparse=False,
)

# ray configurations
tune_config = dict(
    name=__file__.replace(".py", "") + "_3",
    num_samples=10,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 100},
    resources_per_trial={"cpu": 1, "gpu": 0.20},
    loggers=DEFAULT_LOGGERS,
    verbose=0,
)

run_ray(tune_config, base_exp_config)
