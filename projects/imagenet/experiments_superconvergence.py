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

import torch

from projects.imagenet.experiments import CONFIGS, DEFAULT

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

    # dict(start_epoch: image_size)
    # epoch_resize={
    #     0: 128,
    #     14: 224,
    #     32: 288
    # },
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
    # No weigh decay from batch norm modules
    batch_norm_weight_decay=False,
    init_batch_norm=True,
)

FASTAI100 = copy.deepcopy(FASTAI18)
FASTAI100.update(
    epochs=35,
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

    # Create default sparse network
    model_args=dict(config=dict(num_classes=100, defaults_sparse=False)),
)

FASTAI10 = copy.deepcopy(DEFAULT)
FASTAI10.update(
    epochs=50,
    num_classes=10,

    # Create default sparse network
    model_args=dict(config=dict(num_classes=10, defaults_sparse=False)),
)

# Default sparse ResNet-50 for 100 classes
SPARSE100_SUPER = copy.deepcopy(FASTAI100)
SPARSE100_SUPER.update(dict(
    epochs=35,

    # Create default sparse network
    model_args=dict(config=dict(num_classes=100, defaults_sparse=True)),

))


# LR EXPERIMENTS (no momentum)

FASTAI100_HIGHLR = copy.deepcopy(FASTAI100)
FASTAI100_HIGHLR.update(
    lr_scheduler_args=dict(
        # warm-up LR from 1 to 2 for 5 epochs with final LR 0.00025 after 40 epochs
        max_lr=6.0,
        div_factor=6,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=4.0 / 35.0,
        epochs=35,
        anneal_strategy="linear",
    ),
)

SPARSE100_SUPER_HIGHLR = copy.deepcopy(SPARSE100_SUPER)
SPARSE100_SUPER_HIGHLR.update(dict(
    lr_scheduler_args=dict(
        max_lr=6.0,
        div_factor=6,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=4.0 / 35.0,
        epochs=35,
        anneal_strategy="linear",
        base_momentum=0.0,
        max_momentum=0.5,
        cycle_momentum=True,
    ),

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.5,
        nesterov=True,
    ),

))


# NO MOMENTUM EXPERIMENTS

FASTAI100NoMomentum = copy.deepcopy(FASTAI100)
FASTAI100NoMomentum.update(
    lr_scheduler_args=dict(
        max_lr=2.0,
        div_factor=2,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=6.0 / 35.0,
        epochs=35,
        anneal_strategy="linear",
        base_momentum=0.0,
        max_momentum=0.01,
        cycle_momentum=False,
    ),

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.0,
        nesterov=False,
    ),
    num_classes=100,

)

# Default sparse ResNet-50 for 100 classes
SPARSE100_SUPER_SomeMomentum = copy.deepcopy(SPARSE100_SUPER)
SPARSE100_SUPER_SomeMomentum.update(dict(
    lr_scheduler_args=dict(
        max_lr=2.0,
        div_factor=2,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=6.0 / 35.0,
        epochs=35,
        anneal_strategy="linear",
        base_momentum=0.0,
        max_momentum=0.5,
    ),

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.5,
        nesterov=False,
    ),
))

# Default sparse ResNet-50 for 100 classes
SPARSE100_SUPER_NoMomentum = copy.deepcopy(SPARSE100_SUPER)
SPARSE100_SUPER_NoMomentum.update(dict(
    lr_scheduler_args=dict(
        max_lr=2.0,
        div_factor=2,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=6.0 / 35.0,
        epochs=35,
        anneal_strategy="linear",
        base_momentum=0.0,
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
# Default sparse ResNet-50 for 10 classes

DEBUG = copy.deepcopy(FASTAI18)
DEBUG.update(
    num_classes=10,
    model_args=dict(config=dict(num_classes=10)),
)

# Export all configurations
CONFIGS.update(
    dict(

        # Super convergence experiments
        debug=DEBUG,
        super_convergence=SUPER_CONVERGENCE,
        fastai18=FASTAI18,
        fastai100=FASTAI100,
        fastai10=FASTAI10,
        fastai100_no_momentum=FASTAI100NoMomentum,
        fastai100_highlr=FASTAI100_HIGHLR,

        sparse100_super=SPARSE100_SUPER,
        sparse100_super_no_momentum=SPARSE100_SUPER_NoMomentum,
        sparse100_super_some_momentum=SPARSE100_SUPER_SomeMomentum,
        sparse100_super_high_lr=SPARSE100_SUPER_HIGHLR,
    )
)
