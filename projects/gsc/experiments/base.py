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
Base Imagenet Experiment configuration.
"""

import os
import sys
from copy import deepcopy

import torch

from nupic.research.frameworks.pytorch.datasets import preprocessed_gsc
from nupic.research.frameworks.pytorch.models.le_sparse_net import LeSparseNet
from nupic.research.frameworks.vernon import (
    RezeroedKWinnersGSCExperiment,
    VariedRezeroedKWinnersGSCExperiment,
)
from nupic.torch.models.sparse_cnn import gsc_sparse_cnn

# Batch size depends on the GPU memory.
BATCH_SIZE = 16

# Default configuration based on Pytorch GSC training example.
# This is a replicates the config of `sparseCNN2` from
#   `nupic.research\projects\whydense\gsc\experiments_v3.cfg`
# This achieves 94.7073% accuracy
DEFAULT_BASE = dict(
    experiment_class=RezeroedKWinnersGSCExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/gsc"),

    # Dataset
    dataset_class=preprocessed_gsc,
    dataset_args=dict(
        root="~/nta/datasets/gsc_preprocessed",
        download=True,
    ),

    # Seed
    seed=20,

    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.
    num_samples=1,

    # Training batch size
    batch_size=BATCH_SIZE,
    # Validation batch size
    val_batch_size=BATCH_SIZE,
    # Number of batches per epoch. Useful for debugging
    batches_in_epoch=sys.maxsize,

    # Update this to stop training when accuracy reaches the metric value
    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),

    # Number of epochs
    epochs=30,

    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs

    # Model class. Must inherit from "torch.nn.Module"
    model_class=LeSparseNet,

    # Model class arguments passed to the constructor
    model_args=dict(
        input_shape=(1, 32, 32),
        cnn_out_channels=(64, 64),
        cnn_activity_percent_on=(0.095, 0.125),
        cnn_weight_percent_on=(0.5, 0.2),
        linear_n=(1000,),
        linear_activity_percent_on=(0.1,),
        linear_weight_percent_on=(0.1,),
        boost_strength=1.5,
        boost_strength_factor=0.9,
        use_batch_norm=True,
        dropout=0.0,
        num_classes=12,
        k_inference_factor=1.0,
        activation_fct_before_max_pool=True,
        consolidated_sparse_weights=False,
        use_kwinners_local=False,
    ),

    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.SGD,

    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(
        lr=0.01,
        weight_decay=0.01,
        momentum=0.0,
    ),

    # Whether or not to apply weight decay to batch norm modules parameters
    # If False, remove 'weight_decay' from batch norm parameters
    # See https://arxiv.org/abs/1807.11205
    batch_norm_weight_decay=False,

    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,

    # Learning rate scheduler class class arguments passed to the constructor
    lr_scheduler_args=dict(
        gamma=0.9,
        step_size=1,
    ),

    # Loss function. See "torch.nn.functional"
    loss_function=torch.nn.functional.cross_entropy,

    # How often to checkpoint (epochs)
    checkpoint_freq=0,
    keep_checkpoints_num=1,
    checkpoint_score_attr="training_iteration",

    # How many times to try to recover before stopping the trial
    max_failures=3,

    # How many times to retry the epoch before stopping. This is useful when
    # using distributed training with spot instances.
    max_retries=3,

    # Python Logging level : "critical", "error", "warning", "info", "debug"
    log_level="debug",

    # Python Logging Format
    log_format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",

    # Ray tune verbosity. When set to the default value of 2 it will log
    # iteration result dicts. This dict can flood the console if it contains
    # large data structures, so default to verbose=1. The SupervisedTrainable logs
    # a succinct version of the result dict.
    verbose=1,
)


# This config is based off the example from `nupic.torch\examples\gsc\run_gsc_model.py`
# That example model achieves an average of 96.003% acc while the config below
# achieves 96.191% (both averaged over three trials) - so they are comparable.
#
DEFAULT_SPARSE_CNN = deepcopy(DEFAULT_BASE)
DEFAULT_SPARSE_CNN.update(

    # Enable a varied batch size.
    experiment_class=VariedRezeroedKWinnersGSCExperiment,

    # Model
    model_class=gsc_sparse_cnn,
    model_args=dict(),

    # Loss
    loss_function=torch.nn.functional.nll_loss,

    # Batch size
    batch_size=None,
    batch_sizes=[4, 16],  # 4 for the first epoch and 16 for the remaining
    val_batch_size=1000,
    batches_in_epoch=sys.maxsize,
    epochs_to_validate=range(0, 30),
)

# Export configurations in this file
CONFIGS = dict(
    default_base=DEFAULT_BASE,
    default_sparse_cnn=DEFAULT_SPARSE_CNN,
)
