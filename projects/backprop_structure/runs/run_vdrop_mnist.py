# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

"""
Replicate the MNIST result from https://dl.acm.org/doi/10.5555/3305890.3305939
"""

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

NUM_TRAINING_ITERATIONS = 200


class VDropExperiment(mixins.ConstrainParameters,
                      mixins.LogStructure,
                      mixins.Regularize,
                      experiments.Supervised):
    def __init__(self, logdir):

        super().__init__(
            logdir=logdir,

            network_class=networks.mnist_lenet_vdrop,
            network_args=dict(),

            dataset_class=datasets.MNIST,
            dataset_args={},

            training_iterations=NUM_TRAINING_ITERATIONS,

            use_tqdm=False,
            batch_size_test=1000,
            batch_size_train=(64, 64),

            log_verbose_structure=False,

            reg_schedule=dict(
                [(0, 0.0)]
                + list(zip(range(4, 19), np.linspace(0, 1, 15, endpoint=False)))
                + [(19, 1.0)]
            ),
            downscale_reg_with_training_set=True,
        )

    def run_epoch(self, iteration):
        lr = 1e-3 * ((NUM_TRAINING_ITERATIONS - iteration) / NUM_TRAINING_ITERATIONS)
        if iteration == 0 or 5 <= iteration <= 19:
            # Either the optimizer doesn't exist yet, or we're changing the loss
            # function and the adaptive state is invalidated. I don't think the
            # original paper contained this logic, never resetting the Adam
            # optimizer while performing "warmup" on the regularization, but it
            # seems like the right thing to do.
            self.optimizer = torch.optim.Adam(self._get_parameters(),
                                              lr=lr)
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        return super().run_epoch(iteration)


if __name__ == "__main__":
    ray.init()

    tune.run(
        experiments.as_ray_trainable(VDropExperiment),
        name=os.path.basename(__file__).replace(".py", ""),
        num_samples=1,
        checkpoint_freq=0,
        checkpoint_at_end=False,
        resources_per_trial={
            "cpu": 1,
            "gpu": (1 if torch.cuda.is_available() else 0)},
        loggers=DEFAULT_LOGGERS,
        verbose=1,
    )
