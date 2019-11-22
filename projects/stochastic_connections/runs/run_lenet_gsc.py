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
import torch
from ray import tune

import nupic.research.frameworks.stochastic_connections.experiments as experiments
from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)

ray.init()

tune.run(
    experiments.NoiseRay,
    name=os.path.basename(__file__).replace(".py", ""),
    config=dict(
        use_tqdm=False,
        noise_test_epochs=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99],

        dataset_config=dict(
            dataset_name="PreprocessedGSC",
            data_dir=os.path.expanduser("~/nta/datasets/gsc"),
            batch_size_train=16,
            batch_size_test=1000,
            test_noise=True,
            noise_level=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        ),

        model_alg="gsc_lenet",
        model_params=dict(),

        optim_alg="Adam",
        optim_params=dict(
            lr=0.01,
        ),

        lr_scheduler_alg="StepLR",
        lr_scheduler_params=dict(
            step_size=1,
            gamma=0.94,
        ),
    ),
    num_samples=1,
    checkpoint_freq=0,
    checkpoint_at_end=True,
    stop={"training_iteration": 100},
    resources_per_trial={
        "cpu": 1,
        "gpu": (1 if torch.cuda.is_available() else 0)},
    loggers=DEFAULT_LOGGERS,
    verbose=1,
)
