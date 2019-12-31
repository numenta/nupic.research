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

# from ray import tune
import ray
import torch

from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)
from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# experiment configurations
cuda_device_count = torch.cuda.device_count()
base_exp_config = dict(

    # ---- Torch ----
    device="cuda" if cuda_device_count > 0 else "cpu",

    # ---- Dataset ----
    dataset_name="PreprocessedGSC",
    data_dir=os.path.expanduser("~/nta/datasets/gsc"),
    batch_size_train=(4, 16),
    batch_size_test=1000,

    # ---- Network ----
    network=ray.tune.grid_search([
        "gsc_binary_cnn", "gsc_hard_concrete_cnn"
    ]),
    droprate_init=0.2,
    l0_strength=1e-4,

    # ---- Optimizer ----
    optim_alg="Adam",
    learning_rate=0.01,

    # ---- LR Scheduler ----
    lr_scheduler="StepLR",
    lr_step_size=1,
    lr_gamma=0.9825,

    # ---- Model ----
    model=ray.tune.grid_search(["StochasticSynapsesModel"]),
    # debug:
    use_tqdm=True,
    test_noise=False,
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
tune_config = dict(
    name=os.path.basename(__file__).replace(".py", ""),
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 100},
    resources_per_trial={
        # 1 GPU per trial
        "cpu": os.cpu_count() / cuda_device_count,
        "gpu": 1},
    loggers=DEFAULT_LOGGERS,
    verbose=0,
)

run_ray(tune_config, base_exp_config)
