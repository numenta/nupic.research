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

from dynamic_sparse.common.loggers import DEFAULT_LOGGERS
from dynamic_sparse.common.utils import Trainable

torch.manual_seed(32)  # run diverse samples

# alternative initialization based on configuration
exp_config = dict(
    # model related
    device="cuda",
    network="vgg19_bn",
    num_classes=10,
    model="DSNN",
    # model=tune.grid_search(["SET", "DSNN", "DSNN_Flip", "DSNN_Correct"]),
    # dataset related
    dataset_name="CIFAR10",
    augment_images=True,
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir="~/nta/datasets",
    # optimizer related
    optim_alg="SGD",
    momentum=0.9,
    learning_rate=0.01,
    lr_scheduler="MultiStepLR",
    lr_milestones=[250, 290],
    lr_gamma=0.10,
    debug_weights=True,
    # sparse related
    epsilon=tune.grid_search([50, 100, 200]),
    start_sparse=1,
    # end_sparse=-1, # search later
    debug_sparse=True,
    flip=False,  # search later, along with end sparse
    weight_prune_perc=tune.grid_search([0, 0.1, 0.2, 0.3, 0.4]),
    grad_prune_perc=tune.grid_search([0, 0.1, 0.2, 0.3, 0.4]),
)

tune_config = dict(
    name="SET_DSNN_GS1",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    config=exp_config,
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 300},
    resources_per_trial={"cpu": 1, "gpu": 1},
    loggers=DEFAULT_LOGGERS,
    verbose=1,
)

# TODO: automatically counts how many GPUs there are

# override when running local for test
if not torch.cuda.is_available():
    exp_config["device"] = "cpu"
    tune_config["resources_per_trial"] = {"cpu": 1}

# download_dataset(exp_config)
ray.init()
tune.run(Trainable, **tune_config)
