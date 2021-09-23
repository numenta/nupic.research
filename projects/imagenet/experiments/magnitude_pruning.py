# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import copy

import numpy as np
import torch
import torch.nn.init
import torch.optim
from ray import tune

from nupic.research.frameworks.pytorch.models.pretrained_models import resnet50_swsl
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.research.frameworks.pytorch.modules import prunable_conv2d, prunable_linear
from nupic.research.frameworks.vernon.distributed import ImagenetExperiment, mixins

from .base import DEFAULT


class MagPruneExperiment(mixins.LogEveryLoss,
                         mixins.LogEveryLearningRate,
                         mixins.KnowledgeDistillation,
                         mixins.RezeroWeights,
                         mixins.MultiCycleLR,
                         mixins.PruneLowMagnitude,
                         ImagenetExperiment):
    pass


def conv_target_density(in_channels, out_channels, kernel_size):
    """25% dense everywhere except the stem"""
    if kernel_size == 7:
        return 1.0
    else:
        return 0.25


NUM_CLASSES = 1000
SUBSEQUENT_LR_SCHED_ARGS = dict(
    max_lr=6.0,
    pct_start=0.0625,
    anneal_strategy="linear",
    base_momentum=0.6,
    max_momentum=0.75,
    cycle_momentum=True,
    final_div_factor=1000.0
)

MAGPRUNE_BASE = copy.deepcopy(DEFAULT)
MAGPRUNE_BASE.update(dict(
    experiment_class=MagPruneExperiment,
    log_timestep_freq=10,
    num_classes=NUM_CLASSES,
    batch_size=128,
    val_batch_size=128,
    extra_validations_per_epoch=1,
    validate_on_prune=True,

    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    model_class=resnet50,
    model_args=dict(
        num_classes=NUM_CLASSES,
        conv_layer=prunable_conv2d,
        conv_args=dict(
            target_density=conv_target_density,
        ),
        linear_layer=prunable_linear,
        linear_args=dict(
            target_density=0.25,
        ),
    ),

    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(
        lr=6.0 / 6.0,
        momentum=0.75,
        weight_decay=0.0001,
    ),

    lr_scheduler_class=None,
    lr_scheduler_args=None,

    teacher_model_class=[resnet50_swsl],

    init_batch_norm=True,
    use_auto_augment=True,
    checkpoint_at_end=True,

    sync_to_driver=False,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    batch_norm_weight_decay=False,
))

# 0.72368
INITIAL_EPOCHS = 20
NUM_PRUNINGS = 5
EPOCHS_BETWEEN_PRUNINGS = 5
FINAL_EPOCHS = 10
NUM_EPOCHS = (INITIAL_EPOCHS
              + ((NUM_PRUNINGS - 1) * EPOCHS_BETWEEN_PRUNINGS)
              + FINAL_EPOCHS)
MAGPRUNE20_20_10 = copy.deepcopy(MAGPRUNE_BASE)
MAGPRUNE20_20_10.update(dict(
    name="MAGPRUNE20_20_10",
    epochs_to_validate=range(NUM_EPOCHS),
    epochs=NUM_EPOCHS,

    wandb_args=dict(
        project="magnitude-pruning",
        name="20 epochs initial, 20 epochs doing 5 prunings, 10 final",
    ),

    multi_cycle_lr_args=(
        (0, dict(max_lr=6.0,
                 pct_start=0.2,
                 anneal_strategy="linear",
                 base_momentum=0.6,
                 max_momentum=0.75,
                 cycle_momentum=True,
                 div_factor=6.0,
                 final_div_factor=1000.0)),
        (20, SUBSEQUENT_LR_SCHED_ARGS),
        (25, SUBSEQUENT_LR_SCHED_ARGS),
        (30, SUBSEQUENT_LR_SCHED_ARGS),
        (35, SUBSEQUENT_LR_SCHED_ARGS),
        (40, SUBSEQUENT_LR_SCHED_ARGS),
    ),

    prune_schedule=[
        (20, 1 / 5),
        (25, 2 / 5),
        (30, 3 / 5),
        (35, 4 / 5),
        (40, 5 / 5),
    ],
))

# 0.72892
INITIAL_EPOCHS = 15
NUM_PRUNINGS = 6
EPOCHS_BETWEEN_PRUNINGS = 3
FINAL_EPOCHS = 30
NUM_EPOCHS = (INITIAL_EPOCHS
              + ((NUM_PRUNINGS - 1) * EPOCHS_BETWEEN_PRUNINGS)
              + FINAL_EPOCHS)
MAGPRUNE15_15_30 = copy.deepcopy(MAGPRUNE_BASE)
MAGPRUNE15_15_30.update(dict(
    name="MAGPRUNE15_15_30",
    epochs_to_validate=range(NUM_EPOCHS),
    epochs=NUM_EPOCHS,

    optimizer_args=dict(
        lr=6.0 / 6.0,
        momentum=0.75,
        weight_decay=1e-5,
    ),

    wandb_args=dict(
        project="magnitude-pruning",
        name="15-15(6)-30, weight_decay 1e-5",
    ),

    multi_cycle_lr_args=(
        (0, dict(max_lr=6.0,
                 pct_start=0.2,
                 anneal_strategy="linear",
                 base_momentum=0.6,
                 max_momentum=0.75,
                 cycle_momentum=True,
                 div_factor=6.0,
                 final_div_factor=1000.0)),
        (15, SUBSEQUENT_LR_SCHED_ARGS),
        (18, SUBSEQUENT_LR_SCHED_ARGS),
        (21, SUBSEQUENT_LR_SCHED_ARGS),
        (24, SUBSEQUENT_LR_SCHED_ARGS),
        (27, SUBSEQUENT_LR_SCHED_ARGS),
        (30, SUBSEQUENT_LR_SCHED_ARGS),
    ),

    prune_schedule=[
        (15, 1 / 6),
        (18, 2 / 6),
        (21, 3 / 6),
        (24, 4 / 6),
        (27, 5 / 6),
        (30, 6 / 6),
    ],
))


CONFIGS = dict(
    mag_prune_20_20_10=MAGPRUNE20_20_10,
    mag_prune_15_15_30=MAGPRUNE15_15_30,
)
