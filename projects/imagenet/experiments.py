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
import os
import sys

import torch

import nupic.research.frameworks.pytorch.models.resnets


"""
Imagenet Experiment configuration
"""
__all__ = ["CONFIGS"]

# Batch size depends on the GPU memory.
# On AWS P3 (Tesla V100) each GPU can hold 128 batches
BATCH_SIZE = 128

# Default configuration based on Pytorch Imagenet training example.
# See http://github.com/pytorch/examples/blob/master/imagenet/main.py
DEFAULT = dict(
    # Dataset path
    data=os.path.expanduser("~/nta/data/imagenet"),
    # Dataset training data relative path
    train_dir="train",
    # Dataset validation data relative path
    val_dir="val",

    # Results path
    results=os.path.expanduser("~/nta/results/imagenet"),
    # Epoch to start checkpoint
    checkpoint_start=0,

    # Training batch size
    batch_size=BATCH_SIZE,
    # Validation batch size
    val_batch_size=BATCH_SIZE,
    # Number of batches per epoch. Useful for debugging
    batches_in_epoch=sys.maxsize,

    # Stop training when the validation metric reaches the metric value
    stop=dict(mean_accuracy=0.757),
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
    ),

    # Learning rate scheduler class. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    # Learning rate scheduler class class arguments passed to the constructor
    lr_scheduler_args=dict(
        # LR decayed by 10 every 30 epochs
        gamma=0.1,
        step_size=30,
    ),

    # Loss function. See "torch.nn.functional"
    loss_function=torch.nn.functional.cross_entropy,

    # Whether or not to show progress bar during training
    progress_bar=True,

    # Whether or not to use torch.nn.parallel.distributed
    distributed=False,

    # Number of workers used by dataloaders
    workers=0,
)

# Configuration inspired by Fast.ai.
# See https://github.com/fastai/imagenet-fast
FASTAI = dict(DEFAULT)
FASTAI_EPOCHS = 30
FASTAI.update(dict(
    epochs=FASTAI_EPOCHS,
    # Super-Convergence 1cycle policy
    # See https://arxiv.org/pdf/1708.07120.pdf
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
        max_lr=2.0,
        # Total batches per epoch. This is required by the "OneCycleLR" LR Scheduler
        # Total of 1,281,168 images in training dataset
        steps_per_epoch=int(1281168 / BATCH_SIZE),
        epochs=FASTAI_EPOCHS
    ),
))
# Reduce weight decay when using super-convergence
FASTAI["optimizer_args"].update(dict(weight_decay=3e-6))

# Use smaller "imagenette" dataset with same "fastai" configuration
FASTAI_SMALL = dict(FASTAI)
FASTAI_SMALL.update(dict(
    data=os.path.expanduser("~/nta/data/imagenette/sz/160"),
    model_args=dict(config=dict(num_classes=10))))
FASTAI_SMALL["lr_scheduler_args"].update(dict(
    steps_per_epoch=int(12906 / BATCH_SIZE)))

# Export configurations
CONFIGS = dict()
CONFIGS["default"] = DEFAULT
CONFIGS["fastai"] = FASTAI
CONFIGS["fastai_small"] = FASTAI_SMALL
