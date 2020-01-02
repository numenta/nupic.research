#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions appl":
#
#  This program is free softwar": you can redistribute it and/or modify
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
import copy
import os
import sys

import torch

import nupic.research.frameworks.pytorch.models.resnets

"""
Imagenet Experiment configurations
"""
__all__ = ["CONFIGS"]

# Batch size depends on the GPU memory.
# On AWS P3 (Tesla V100) each GPU can hold 128 batches
BATCH_SIZE = 128

# Default configuration based on Pytorch Imagenet training example.
# See http://github.com/pytorch/examples/blob/master/imagenet/main.py
DEFAULT = dict(
    # Results path
    local_dir=os.path.expanduser("~/nta/results/imagenet"),
    # Dataset path
    data=os.path.expanduser("~/nta/data/imagenet"),
    # Dataset training data relative path
    train_dir="train",
    # Dataset validation data relative path
    val_dir="val",
    # Limit the dataset size to the given number of classes
    num_classes=1000,

    # Training batch size
    batch_size=BATCH_SIZE,
    # Validation batch size
    val_batch_size=BATCH_SIZE,
    # Number of batches per epoch. Useful for debugging
    batches_in_epoch=sys.maxsize,

    # Stop training when the validation metric reaches the metric value
    stop=dict(mean_accuracy=0.85),
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

    # Progressive resize schedule - dict(start_epoch: image_size)
    # See https://arxiv.org/pdf/1806.01427.pdf
    # epoch_resize={0: 224},

    # Loss function. See "torch.nn.functional"
    loss_function=torch.nn.functional.cross_entropy,

    # How often to checkpoint (epochs)
    checkpoint_freq=0,
    # How many times to try to recover before stopping the trial
    max_failures=-1,
)

DEFAULT10 = copy.deepcopy(DEFAULT)
DEFAULT10.update(
    epochs=100,
    num_classes=10,
    init_batch_norm=False,

    # Create default sparse network
    model_args=dict(config=dict(num_classes=10, defaults_sparse=False)),
)

# Use normal schedule
SPARSE10 = copy.deepcopy(DEFAULT10)
SPARSE10.update(
    # Create default sparse network
    model_args=dict(config=dict(num_classes=10, defaults_sparse=True)),
)

DEFAULT100 = copy.deepcopy(DEFAULT)
DEFAULT100.update(
    epochs=100,
    num_classes=100,
    init_batch_norm=False,

    # Create default sparse network
    model_args=dict(config=dict(num_classes=100, defaults_sparse=False)),
)

# Use normal schedule
SPARSE100 = copy.deepcopy(DEFAULT100)
SPARSE100.update(
    # Create default sparse network
    model_args=dict(config=dict(num_classes=100, defaults_sparse=True)),
)

# Use plain learning schedule, i.e. with no momentum
SPARSE100_PLAIN_LEARNING = copy.deepcopy(SPARSE100)
SPARSE100_PLAIN_LEARNING.update(
    optimizer_args=dict(
        lr=0.1,
        weight_decay=1e-04,
        momentum=0.0,
        dampening=0,
        nesterov=False
    ),
    # Create default sparse network
    model_args=dict(config=dict(num_classes=100, defaults_sparse=True)),
)


# Export all configurations
CONFIGS = dict(
    default10=DEFAULT10,
    default_sparse_10=SPARSE10,
    default100=DEFAULT100,
    sparse_100=SPARSE100,
    sparse100_plain_learning=SPARSE100_PLAIN_LEARNING,
)
