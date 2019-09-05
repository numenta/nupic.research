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

from dynamic_sparse.common import DEFAULT_LOGGERS
from nupic.research.frameworks.dynamic_sparse.common.utils import Trainable, download_dataset

torch.manual_seed(32)  # run diverse samples

# alternative initialization based on configuration
exp_config = dict(
    network="vgg19_bn",
    num_classes=10,
    model=tune.grid_search(["BaseModel", "SparseModel", "SET", "DSNN"]),
    # model="DSNN",
    epsilon=60,
    start_sparse=1,
    momentum=0.9,
    learning_rate=0.01,
    lr_scheduler="MultiStepLR",
    lr_milestones=[250, 290],
    lr_gamma=0.10,
    dataset_name="CIFAR10",
    augment_images=True,
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir="~/nta/datasets",
    device="cuda",
    optim_alg="SGD",
    debug_weights=True,
    debug_sparse=True,
)

tune_config = dict(
    name="SET_DSNN_Test",
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

# override when running local for test
if not torch.cuda.is_available():
    exp_config["device"] = "cpu"
    tune_config["resources_per_trial"] = {"cpu": 1}

download_dataset(exp_config)
ray.init()
tune.run(Trainable, **tune_config)
