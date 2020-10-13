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
Run a simple GSC experiment using OneCycleLR. The parameters used here
have been optimized by OneCycleLr, and achieved an accuracy of 0.964
averaged over 5 model examples.
"""

from copy import deepcopy

import torch

from .base import DEFAULT_BASE

SPARSE_CNN_ONECYCLELR = deepcopy(DEFAULT_BASE)
SPARSE_CNN_ONECYCLELR.update(
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,

    lr_scheduler_args=dict(
        max_lr=0.11, #Optimized in Sig-Opt
        div_factor=50,  
        final_div_factor=4000, 
        pct_start=0.15,
        epochs=30,
        anneal_strategy="linear",
        max_momentum=0.01,
        cycle_momentum=False,
    ),

    epochs=30,
    epochs_to_validate=range(0, 30),

    # Training batch size
    batch_size=128,
    # Validation batch size
    val_batch_size=128,

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.0,
        nesterov=False,
    ),
)


CONFIGS = dict(
    sparse_cnn_onecyclelr=SPARSE_CNN_ONECYCLELR,
)
