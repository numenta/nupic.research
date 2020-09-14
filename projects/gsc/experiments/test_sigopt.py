#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Test SigOpt hyper-parameter tuning with GSC Experiment configuration.
"""

from copy import deepcopy

import torch

from nupic.research.frameworks.sigopt import SigOptSGDStepLRExperiment

from .base import DEFAULT_SPARSE_CNN

SIGOPT_SPARSE_CNN_STEPLR = deepcopy(DEFAULT_SPARSE_CNN)
SIGOPT_SPARSE_CNN_STEPLR.update(
    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,

    # Learning rate scheduler class class arguments passed to the constructor
    lr_scheduler_args=dict(
        gamma=0.9,
        step_size=1,
    ),

    sigopt_experiment_class=SigOptSGDStepLRExperiment,

    epochs=30,
    epochs_to_validate=range(0, 30),

    sigopt_config=dict(
        name="experiment-name",
        # See https://app.sigopt.com/docs/overview/parameter_bounds
        parameters=[
            dict(name="gamma", type="double", bounds=dict(min=0.01, max=0.99)),
            dict(name="step_size", type="double", bounds=dict(min=1, max=30)),
        ],
        metrics=[dict(name="mean_accuracy", objective="maximize")],
        parallel_bandwidth=1,
        observation_budget=10,
        project="mcaporale-examples",
    ),

    sigopt_experiment_id=325853,

)


CONFIGS = dict(
    sigopt_sparse_cnn_steplr=SIGOPT_SPARSE_CNN_STEPLR,
)
