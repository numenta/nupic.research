#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import torch

from projects.imagenet.experiments import CONFIGS, DEFAULT
from projects.imagenet.experiments_custom_super import CUSTOM_CONFIG

"""
Imagenet Experiment configurations
"""
__all__ = ["CONFIGS"]


# SUPER CONVERGENCE EXPERIMENTS

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

        # Determines the initial learning rate via initial_lr = max_lr/div_factor
        div_factor=20,  # initial_lr = 0.05

        # Determines the minimum learning rate via min_lr = initial_lr/final_div_factor
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
    # No weight decay from batch norm modules
    batch_norm_weight_decay=False,
    init_batch_norm=True,
)

PROGRESSIVE_RESIZE = copy.deepcopy(FASTAI18)
PROGRESSIVE_RESIZE.update(
    # Progressive image resize schedule - dict(start_epoch: image_size)
    # See:
    # - https://arxiv.org/pdf/1806.01427.pdf
    # - https://arxiv.org/abs/1707.02921
    # - https://arxiv.org/abs/1710.10196
    train_dir="sz/160/train",
    progressive_resize={
        "0": 128,
        "20": 224,
        "35": 288,
    },
    # Works with progressive_resize and the available GPU memory to fit as many
    # images as possible in each batch - dict(start_epoch: batch_size)
    dynamic_batch_size={
        "0": 512,
        "20": 160,
        "35": 96,
    },
    lr_scheduler_args=dict(
        # warm-up LR from 1 to 2 for 5 epochs with final LR 0.00025 after 35 epochs
        max_lr=2.0,
        div_factor=2,  # initial_lr = 1.0
        # Scale final LR by final batch_size to initial batch_size ratio
        final_div_factor=96 / 512 / 0.00025,  # min_lr = 0.00005
        pct_start=5.0 / 35.0,
        anneal_strategy="linear",
    ),
)

FASTAI100 = copy.deepcopy(FASTAI18)
FASTAI100.update(
    epochs=35,
    log_level="debug",
    lr_scheduler_args=dict(
        # warm-up LR from 1 to 2 for 5 epochs with final LR 0.00025 after 40 epochs
        max_lr=2.0,
        div_factor=2,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=6.0 / 35.0,
        epochs=35,
        anneal_strategy="linear",
    ),
    num_classes=100,
    model_args=dict(config=dict(num_classes=100, defaults_sparse=False)),
)

FASTAI1000 = copy.deepcopy(FASTAI100)
FASTAI1000.update(
    epochs=35,
    num_classes=1000,

    model_args=dict(config=dict(num_classes=1000, defaults_sparse=False)),
)

# This gets to about 78.6% after 35 epochs
SPARSE100_SUPER = copy.deepcopy(FASTAI100)
SPARSE100_SUPER.update(dict(
    epochs=35,
    log_level="debug",

    model_args=dict(config=dict(num_classes=100, defaults_sparse=True)),

    # Use a higher learning rate and no momentum for sparse superconvergence
    lr_scheduler_args=dict(
        max_lr=6.0,
        div_factor=6,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=4.0 / 35.0,
        epochs=35,
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

))


# Default sparse ResNet-50 for 100 classes
SPARSE1000_SUPER = copy.deepcopy(SPARSE100_SUPER)
SPARSE1000_SUPER.update(dict(
    epochs=35,
    log_level="debug",
    num_classes=1000,

    model_args=dict(config=dict(num_classes=1000, defaults_sparse=True)),

))

DEBUG18 = copy.deepcopy(FASTAI18)
DEBUG18.update(
    num_classes=10,
    model_args=dict(config=dict(num_classes=10)),
)

# Export all configurations
CONFIGS.update(CUSTOM_CONFIG)
CONFIGS.update(
    dict(

        # Super convergence experiments
        debug_fastai18=DEBUG18,
        super_convergence=SUPER_CONVERGENCE,
        fastai18=FASTAI18,
        progressive_resize=PROGRESSIVE_RESIZE,
        fastai100=FASTAI100,
        fastai1000=FASTAI1000,

        sparse100_super=SPARSE100_SUPER,
        sparse1000_super=SPARSE1000_SUPER,
    )
)
