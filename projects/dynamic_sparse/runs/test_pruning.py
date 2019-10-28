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

from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray
from ray import tune
import numpy as np

# alternative initialization based on configuration
exp_config = dict(
    device="cuda",
    network="resnet18",
    dataset_name="CIFAR10",
    input_size=(3,32,32),
    augment_images=True,
    num_classes=10,
    model="PruningModel",
    # specific pruning
    target_final_density=0.1,
    start_pruning_epoch=2,
    end_pruning_epoch=150,
    epochs=200,
    # sparsity related
    on_perc=tune.grid_search(list(np.arange(0.1,1.01, 0.05))),
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
    name=__file__.replace(".py", "") + "3",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    resources_per_trial={"cpu": 1, "gpu": .245},
    verbose=2,
)

run_ray(tune_config, exp_config, fix_seed=True)
