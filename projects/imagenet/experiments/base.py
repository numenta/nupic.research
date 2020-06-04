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

import copy
import os
import sys

import ray.tune as tune
import torch

import nupic.research.frameworks.pytorch.models.resnets
from nupic.research.frameworks.pytorch.imagenet import (
    RezeroedKWinnersImagenetExperiment,
)

# Batch size depends on the GPU memory.
# On AWS P3 (Tesla V100) each GPU can hold 128 batches
BATCH_SIZE = 128

# Default configuration based on Pytorch Imagenet training example.
# See http://github.com/pytorch/examples/blob/master/imagenet/main.py
DEFAULT = dict(
    experiment_class=RezeroedKWinnersImagenetExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/imagenet"),
    # Dataset location (directory path or HDF5 file with the raw images)
    data=os.path.expanduser("~/nta/data/imagenet/imagenet.hdf5"),
    # Dataset training data relative path
    train_dir="train",
    # Dataset validation data relative path
    val_dir="val",
    # Limit the dataset size to the given number of classes
    num_classes=1000,

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
    epochs=90,

    # Model class. Must inherit from "torch.nn.Module"
    model_class=nupic.research.frameworks.pytorch.models.resnets.resnet50,
    # model model class arguments passed to the constructor
    model_args=dict(),

    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.SGD,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(
        lr=0.1,
        weight_decay=1e-04,
        momentum=0.9,
        dampening=0,
        nesterov=True
    ),
    # Whether or not to apply weight decay to batch norm modules parameters
    # If False, remove 'weight_decay' from batch norm parameters
    # See https://arxiv.org/abs/1807.11205
    batch_norm_weight_decay=True,

    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    # Learning rate scheduler class class arguments passed to the constructor
    lr_scheduler_args=dict(
        # LR decayed by 10 every 30 epochs
        gamma=0.1,
        step_size=30,
    ),

    # Whether or not to Initialize running batch norm mean to 0.
    # See https://arxiv.org/pdf/1706.02677.pdf
    init_batch_norm=False,

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
    # large data structures, so default to verbose=1. The ImagenetTrainable logs
    # a succinct version of the result dict.
    verbose=1,
)

DEBUG = copy.deepcopy(DEFAULT)
DEBUG.update(
    epochs=5,
    num_classes=3,
    model_args=dict(config=dict(num_classes=3, defaults_sparse=False)),

    seed=tune.grid_search([42, 43])

)

DEBUG_SPARSE = copy.deepcopy(DEFAULT)
DEBUG_SPARSE.update(
    epochs=5,
    num_classes=3,
    model_args=dict(config=dict(num_classes=3, defaults_sparse=True)),

    seed=tune.grid_search([42, 43])

)


# Export configurations in this file
CONFIGS = dict(
    default_base=DEFAULT,
    debug_base=DEBUG,
    debug_base_sparse=DEBUG_SPARSE,
)
