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
    stop=dict(mean_accuracy=0.75),
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
    epoch_resize={0: 224},

    # Loss function. See "torch.nn.functional"
    loss_function=torch.nn.functional.cross_entropy,

    # How often to checkpoint (epochs)
    checkpoint_freq=1,
)

# Configuration inspired by Super-Convergence paper. (Fig 6a)
# See https://arxiv.org/pdf/1708.07120.pdf
# 1cycle learning rate policy with the learning rate varying from 0.05 to 1.0,
# then down to 0.00005 in 20 epochs, using a weight decay of 3e−6
SUPER_CONVERGENCE = copy.deepcopy(DEFAULT)
SUPER_CONVERGENCE.update(
    epochs=20,
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
        max_lr=1.0,
        div_factor=20,  # initial_lr = 0.05
        final_div_factor=1000,  # min_lr = 0.00005
        anneal_strategy="linear",
    ),
    # Reduce weight decay to 3e-6 when using super-convergence
    optimizer_args=dict(
        lr=0.1,
        weight_decay=3e-6,
        momentum=0.9,
        nesterov=False,
    ),
)

# Configuration inspired by https://www.fast.ai/2018/08/10/fastai-diu-imagenet/
# https://app.wandb.ai/yaroslavvb/imagenet18/runs/gxsdo6i0
FASTAI18 = copy.deepcopy(SUPER_CONVERGENCE)
FASTAI18.update(
    epochs=35,

    # dict(start_epoch: image_size)
    epoch_resize={
        0: 128,
        14: 224,
        32: 288
    },
    lr_scheduler_args=dict(
        # warm-up LR from 1 to 2 for 5 epochs with final LR 0.00025 after 35 epochs
        max_lr=2.0,
        div_factor=2,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=5.0 / 35.0,
        anneal_strategy="linear",
    ),
    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.9,
        nesterov=False,
    ),

    # Initialize running batch norm mean to 0
    # See https://arxiv.org/pdf/1706.02677.pdf
    init_batch_norm=True,

    # Remove weight decay to batch norm modules parameters
    # See https://arxiv.org/abs/1807.11205
    batch_norm_weight_decay=False,

)

DEBUG = copy.deepcopy(FASTAI18)
DEBUG.update(
    num_classes=10,
    model_args=dict(config=dict(num_classes=10)),
)

# Export all configurations
CONFIGS = dict(
    default=DEFAULT,
    debug=DEBUG,
    super_convergence=SUPER_CONVERGENCE,
    fastai18=FASTAI18,
)
