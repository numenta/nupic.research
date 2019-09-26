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
from ray import tune

from nupic.research.frameworks.dynamic_sparse.common.loggers import DEFAULT_LOGGERS
from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# define a small convolutional network 
# in line with the 

# experiment configurations
base_exp_config = dict(
    device="cuda",
    # ----- dataset related ----
    dataset_name="PreprocessedGSC",
    data_dir=os.path.expanduser("~/nta/datasets/gsc"),
    train_batches_per_epoch=5121,   
    # batch_size_train=(4, 16),
    batch_size_train=16,
    test_batches_per_epoch=50, # total of 1000 samples, similar to HSD
    batch_size_test=20, # required to fit the GPU
    # ----- network related ----
    network="GSCHeb",
    percent_on_k_winner=[0.095, 0.125, 0.067],
    k_inference_factor=1.5,
    boost_strength = 1.5,
    boost_strength_factor = 0.9,
    hidden_neurons_conv=[64, 64],
    hidden_neurons_fc=1500,    
    bias=True,
    dropout=False,
    batch_norm=True,
    # ----- model related ----
    model=tune.grid_search(["BaseModel", "SparseModel", "DSNNWeightedMag", "DSNNMixedHeb"]),
    optim_alg="SGD",
    momentum=0, 
    learning_rate=0.01,
    weight_decay= 0.01,
    lr_scheduler="StepLR",
    lr_gamma=0.9,
    on_perc=[1 , 1, 0.1, 1],
    hebbian_prune_perc=None,
    hebbian_grow=False,
    weight_prune_perc=0.3,
    pruning_early_stop=None, # 2
    # additional validation
    test_noise=True,
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
tune_config = dict(
    name=__file__.replace(".py", "") + "_test1",
    num_samples=5,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 25},
    resources_per_trial={"cpu": 1, "gpu": .20},
    loggers=DEFAULT_LOGGERS,
    verbose=0,
)

run_ray(tune_config, base_exp_config)

