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

import os

import numpy as np
import torch.optim

import nupic.research.frameworks.backprop_structure.dataset_managers as datasets
import nupic.research.frameworks.backprop_structure.experiments as experiments
import nupic.research.frameworks.backprop_structure.experiments.mixins as mixins
import nupic.research.frameworks.backprop_structure.networks as networks
from nupic.research.frameworks.backprop_structure.ray_ax import ax_optimize_accuracy

NUM_TRAINING_ITERATIONS = 30

PARAMETERS = [
    {"name": "lr", "type": "range", "bounds": [0.00001, 0.3],
     "value_type": "float", "log_scale": True},
    {"name": "momentum", "type": "range", "bounds": [0, 0.99999],
     "value_type": "float"},
    {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-1],
     "value_type": "float", "log_scale": True},
    {"name": "gamma", "type": "range", "bounds": [0.3, 0.97],
     "value_type": "float"},
    {"name": "step_size", "type": "range", "bounds": [1, 10],
     "value_type": "int"},
    {"name": "first_batch_size", "type": "range", "bounds": [4, 64],
     "value_type": "int", "log_scale": True},
]


class ExploratoryExperiment(experiments.Supervised):
    def __init__(self, logdir, lr, momentum, weight_decay, gamma, step_size,
                 first_batch_size):
        step_size = int(step_size)
        first_batch_size = int(first_batch_size)

        super().__init__(
            logdir=logdir,

            network_name=networks.mnist_lesparsenet,
            network_args=dict(
                cnn_activity_percent_on=(1.0, 1.0),
                cnn_weight_percent_on=(1.0, 1.0),
                linear_activity_percent_on=(1.0,),
                linear_weight_percent_on=(1.0,),
                use_batch_norm=False,
            ),

            dataset_name=datasets.MNIST,
            dataset_args={},

            optim_alg=torch.optim.SGD,
            optim_args=dict(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args=dict(
                step_size=step_size,
                gamma=gamma,
            ),

            training_iterations=NUM_TRAINING_ITERATIONS,

            use_tqdm=False,
            batch_size_test=1000,
            batch_size_train=(first_batch_size, 64),
        )


class FollowupExperiment(mixins.TestNoise, ExploratoryExperiment):
    def __init__(self, **config):
        super().__init__(noise_test_at_end=True,
                         noise_test_freq=0,
                         noise_levels=list(np.arange(0.0, 0.51, 0.05)),
                         **config)


if __name__ == "__main__":
    experiment_name = os.path.basename(__file__).replace(".py", "")

    num_best_config_samples = 5

    ax_optimize_accuracy(
        experiments.as_ray_trainable(ExploratoryExperiment),
        experiments.as_ray_trainable(FollowupExperiment),
        experiment_name,
        os.path.dirname(os.path.realpath(__file__)),
        PARAMETERS,
        NUM_TRAINING_ITERATIONS,
        num_best_config_samples,
    )
