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

from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)
from nupic.research.frameworks.dynamic_sparse.common.utils import (
    Trainable,
    download_dataset,
)

# experiment configurations
exp_config = dict(
    device="cuda",
    # dataset related
    dataset_name="CIFAR10",
    input_size=(3, 32, 32),
    num_classes=10,
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir="~/nta/datasets",
    augment_images=tune.grid_search([True, False]),
    # model related
    model=tune.grid_search(["DynamicRep", "SparseModel", "BaseModel"]),
    # model="SparseModel",
    network="Wide_ResNet",
    dropout_rate=0,
    depth=28,
    widen_factor=2,
    # optimizer related
    optim_alg="SGD",
    momentum=0.9,
    learning_rate=0.1,
    weight_decay=5e-4,
    lr_scheduler="MultiStepLR",
    lr_milestones=[60, 120, 160],  # 2e-2, 4e-3, 8-e4
    lr_gamma=0.20,
    # sparse related
    on_perc=0.2,
    zeta=0.2,
    start_sparse=1,
    end_sparse=None,
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
tune_config = dict(
    name="wideresnet-test",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 200},  # 300 in cifar
    resources_per_trial={"cpu": 1, "gpu": 1},
    loggers=DEFAULT_LOGGERS,
    verbose=1,
    config=exp_config,
)

# override when running local for test
if not torch.cuda.is_available():
    exp_config["device"] = "cpu"
    tune_config["resources_per_trial"] = {"cpu": 1}

download_dataset(exp_config)
ray.init()
tune.run(Trainable, **tune_config)
