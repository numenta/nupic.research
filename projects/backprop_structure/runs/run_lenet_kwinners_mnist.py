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
import torch.optim
from ray import tune

import nupic.research.frameworks.backprop_structure.dataset_managers as datasets
import nupic.research.frameworks.backprop_structure.experiments as experiments
import nupic.research.frameworks.backprop_structure.experiments.mixins as mixins
import nupic.research.frameworks.backprop_structure.networks as networks
from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)


class SupervisedNoiseBoostingCovariance(mixins.UpdateBoostStrength,
                                        mixins.TestNoise,
                                        mixins.LogCovariance,
                                        experiments.Supervised):
    pass


if __name__ == "__main__":
    ray.init()

    tune.run(
        experiments.as_ray_trainable(SupervisedNoiseBoostingCovariance),
        name=os.path.basename(__file__).replace(".py", ""),
        config=dict(
            network_class=networks.mnist_lesparsenet,
            network_args=dict(
                cnn_weight_percent_on=(1.0, 1.0),
                linear_weight_percent_on=(1.0,),
            ),

            dataset_class=datasets.MNIST,
            dataset_args={},

            optim_class=torch.optim.SGD,
            optim_args=dict(
                lr=0.02,
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args=dict(
                step_size=1,
                gamma=0.8,
            ),

            training_iterations=15,

            use_tqdm=False,
            batch_size_train=(4, 64),
            batch_size_test=1000,

            noise_test_at_end=True,
            noise_test_freq=0,
            noise_levels=list(np.arange(0.0, 1.0, 0.05)),

            log_covariance_layernames=["linear1_kwinners"],
        ),
        num_samples=4,
        checkpoint_freq=0,
        checkpoint_at_end=True,
        resources_per_trial={
            "cpu": 1,
            "gpu": (1 if torch.cuda.is_available() else 0)},
        loggers=DEFAULT_LOGGERS,
        verbose=1,
    )
