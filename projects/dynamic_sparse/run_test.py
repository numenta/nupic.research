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

from utils import Trainable

# alternative initialization based on configuration
config = dict(
    network="resnet18",
    num_classes=10,
    model="DSNN",
    dataset_name="CIFAR10",
    stats_mean=(0.4914, 0.4822, 0.4465),
    stats_std=(0.2023, 0.1994, 0.2010),
    data_dir="~/nta/datasets",
    device="cpu",
    optim_alg="SGD",
    batch_size_train=10,
    batch_size_test=10,
    debug_sparse=False,
)

# run
ray.init()
tune.run(
    Trainable,
    name="SET_local_test",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    config=config,
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 10},
    resources_per_trial={"cpu": 1, "gpu": 0},
)
