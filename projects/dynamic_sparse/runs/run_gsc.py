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

import ray.tune as tune
import torch

from nupic.research.frameworks.dynamic_sparse.common.loggers import DEFAULT_LOGGERS
from nupic.research.frameworks.dynamic_sparse.common.utils import Trainable, init_ray

torch.manual_seed(32)


# experiment configurations
base_exp_config = dict(
    device=("cuda" if torch.cuda.device_count() > 0 else "cpu"),
    dataset_name="PreprocessedGSC",
    data_dir="~/nta/datasets/gsc",
    batch_size_train=(4, 16),
    batch_size_test=1000,
    # ----- Network Related ------
    # SE
    # model=tune.grid_search(["BaseModel", "SparseModel",
    # "DSNNMixedHeb", "DSNNConvHeb"]),
    model="DSNNConvOnlyHeb",
    # model="DSNNConvHeb",
    # network="GSCHeb",
    network="gsc_conv_heb",
    # network="gsc_conv_only_heb",
    # ----- Optimizer Related ----
    optim_alg="SGD",
    momentum=0,
    learning_rate=0.01,
    weight_decay=0.01,
    lr_scheduler="StepLR",
    lr_gamma=0.90,
    use_kwinners=True,
    # use_kwinners=tune.grid_search([True, False]),
    # ----- Dynamic-Sparse Related  - FC LAYER -----
    epsilon=184.61538 / 3,  # 0.1 in the 1600-1000 linear layer
    sparse_linear_only=True,
    start_sparse=1,
    end_sparse=-1,  # don't get last layer
    weight_prune_perc=0.15,
    hebbian_prune_perc=0.60,
    pruning_es=True,
    pruning_es_patience=0,
    pruning_es_window_size=5,
    pruning_es_threshold=0.02,
    pruning_interval=1,
    # ----- Dynamic-Sparse Related  - CONV -----
    prune_methods=tune.grid_search([None, "random", "dynamic", "static"]),
    hebbian_prune_frac=[1.0, 1.0],
    magnitude_prune_frac=[0.0, 0.0],
    sparsity=[0.90, 0.90],
    update_nsteps=[3000, 3000],
    prune_dims=tuple(),
    # ----- Additional Validation -----
    test_noise=False,
    noise_level=0.1,
    # ----- Debugging -----
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
experiment_name = "gsc-test"
tune_config = dict(
    name=experiment_name,
    num_samples=1,
    local_dir=os.path.expanduser(os.path.join("~/nta/results", experiment_name)),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 100},
    resources_per_trial={"cpu": 1, "gpu": 1},
    loggers=DEFAULT_LOGGERS,
    verbose=1,
    config=base_exp_config,
)

# override when running local for test
if not torch.cuda.is_available():
    base_exp_config["device"] = "cpu"
    tune_config["resources_per_trial"] = {"cpu": 1}

init_ray()
tune.run(Trainable, **tune_config)
