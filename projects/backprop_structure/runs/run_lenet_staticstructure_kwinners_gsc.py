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

import numpy as np
import ray
import torch
from ray import tune

import nupic.research.frameworks.backprop_structure.experiments as experiments
import nupic.research.frameworks.backprop_structure.experiments.mixins as mixins
from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)

NUM_TRAINING_ITERATIONS = 30


class SupervisedNoiseBoostingRezero(mixins.UpdateBoostStrength,
                                    mixins.RezeroWeights,
                                    mixins.TestNoise,
                                    experiments.Supervised):
    pass


if __name__ == "__main__":
    ray.init()

    tune.run(
        experiments.as_ray_trainable(SupervisedNoiseBoostingRezero),
        name=os.path.basename(__file__).replace(".py", ""),
        config=dict(
            network_name="gsc_lesparsenet",
            network_params=dict(),

            dataset_name="PreprocessedGSC",
            dataset_params={},

            optim_alg="SGD",
            optim_params=dict(
                lr=0.01,
                weight_decay=0.01,
            ),

            lr_scheduler_alg="StepLR",
            lr_scheduler_params=dict(
                step_size=1,
                gamma=0.9,
            ),

            training_iterations=NUM_TRAINING_ITERATIONS,

            use_tqdm=False,
            batch_size_train=(16, 16),
            batch_size_test=1000,

            noise_test_at_end=True,
            noise_test_freq=0,
            noise_levels=list(np.arange(0.0, 0.51, 0.05)),
        ),
        num_samples=8,
        checkpoint_freq=0,
        checkpoint_at_end=False,
        resources_per_trial={
            "cpu": 1,
            "gpu": (1 if torch.cuda.is_available() else 0)},
        loggers=DEFAULT_LOGGERS,
        verbose=1,
    )
