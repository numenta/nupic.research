#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Run a GSC experiment using OneCycleLR and optimize with SigOpt
"""

from copy import deepcopy

import torch

from nupic.research.frameworks.sigopt import SigOptSGDOneCycleLRExperiment

from .base import DEFAULT_SPARSE_CNN

SIGOPT_SPARSE_CNN_ONECYCLELR = deepcopy(DEFAULT_SPARSE_CNN)
SIGOPT_SPARSE_CNN_ONECYCLELR.update(
    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,

    lr_scheduler_args=dict(
        # commented arguments are suggested by sigopt
        # max_lr=6.91,
        div_factor=10.0, 
        final_div_factor=4000,  
        pct_start=0.1,
        epochs=30,
        anneal_strategy="linear",
        max_momentum=0.01,
        cycle_momentum=False,
    ),

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.0,
        nesterov=False,
    ),

    sigopt_experiment_class=SigOptSGDOneCycleLRExperiment,

    epochs=30,
    epochs_to_validate=range(0, 30),

    sigopt_config=dict(
        name="gsc_onecyclelr_v1",
        # See https://app.sigopt.com/docs/overview/parameter_bounds
        parameters=[
            dict(name="max_lr", type="double", bounds=dict(min=1, max=10)),
        ],
        metrics=[dict(name="mean_accuracy", objective="maximize")],
        parallel_bandwidth=1,
        observation_budget=10,
        project="gsc_basic",
    ),

    sigopt_experiment_id=None,

)


CONFIGS = dict(
    sigopt_sparse_cnn_onecyclelr=SIGOPT_SPARSE_CNN_ONECYCLELR,
)
