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


class SupervisedNoiseConstrainedLoggedRegularized(mixins.TestNoise,
                                                  mixins.ConstrainParameters,
                                                  mixins.LogStructure,
                                                  mixins.Regularize,
                                                  experiments.Supervised):
    pass
    # def run_epoch(self, iteration):
    #     # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     pr = cProfile.Profile()
    #     pr.enable()
    #     result = super().run_epoch(iteration)
    #     pr.disable()
    #     prof.export_chrome_trace(os.path.expanduser("~/chrome-trace{}.trace".format(iteration)))
    #     pstats.Stats(pr).dump_stats(os.path.expanduser("~/now{}.profile".format(iteration)))
    #     return result


if __name__ == "__main__":
    ray.init()

    regimes = [
        dict(optim_alg="Adam",
             optim_params=dict(
                 lr=0.01,
             ),
             lr_scheduler_alg="StepLR",
             lr_scheduler_params=dict(
                 step_size=5,
                 gamma=0.5,
             ),
             training_iterations=50),
    ]

    tune.run(
        experiments.as_ray_trainable(
            SupervisedNoiseConstrainedLoggedRegularized),
        name=os.path.basename(__file__).replace(".py", ""),
        config=tune.grid_search([
            dict(model_alg="mnist_lenet_backpropstructure",
                 model_params=dict(
                     l0_strength=7e-6,
                     droprate_init=0.2,
                 ),

                 dataset_name="MNIST",
                 dataset_params={},

                 use_tqdm=False,
                 batch_size_train=(64, 64),
                 batch_size_test=1000,

                 noise_test_freq=0,
                 noise_test_at_end=True,
                 noise_levels=list(np.arange(0.0, 1.0, 0.05)),
                 **regime)
            for regime in regimes]),
        num_samples=1,
        checkpoint_freq=0,
        checkpoint_at_end=True,
        resources_per_trial={
            "cpu": 1,
            "gpu": (1 if torch.cuda.is_available() else 0)},
        loggers=DEFAULT_LOGGERS,
        verbose=1,
    )
