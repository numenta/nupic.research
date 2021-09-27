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

from nupic.research.frameworks.backprop_structure.model_conversion import (
    maskedvdrop_to_sparseweights,
)
from nupic.research.frameworks.backprop_structure.modules import (
    MaskedVDropCentralData,
    prunable_vdrop_conv2d,
    prunable_vdrop_linear,
)
from nupic.research.frameworks.backprop_structure.networks import vdrop_resnet50
from nupic.research.frameworks.pytorch.models.pretrained_models import resnet50_swsl
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.research.frameworks.pytorch.modules import sparse_conv2d, sparse_linear
from nupic.research.frameworks.vernon.distributed import ImagenetExperiment, mixins

from .base import DEFAULT


class SNPruneExperiment(mixins.LogEveryLoss,
                        mixins.LogEveryLearningRate,
                        mixins.KnowledgeDistillation,
                        mixins.MultiCycleLR,
                        mixins.PruneLowSNRLayers,
                        mixins.RegularizeLoss,
                        mixins.ConstrainParameters,
                        ImagenetExperiment):
    pass


def conv_target_density(in_channels, out_channels, kernel_size):
    """25% dense everywhere except the stem"""
    if kernel_size == 7:
        return 1.0
    else:
        return 0.25


def make_reg_schedule(epochs, pct_ramp_start, pct_ramp_end, peak_value,
                      pct_drop, final_value):
    def reg_schedule(epoch, batch_idx, steps_per_epoch):
        pct = (epoch + batch_idx / steps_per_epoch) / epochs

        if pct < pct_ramp_start:
            return 0.0
        elif pct < pct_ramp_end:
            progress = (pct - pct_ramp_start) / (pct_ramp_end - pct_ramp_start)
            return progress * peak_value
        elif pct < pct_drop:
            return peak_value
        else:
            return final_value

    return reg_schedule


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


SNR_PRUNE_BASE = copy.deepcopy(DEFAULT)
SNR_PRUNE_BASE.update(dict(
    experiment_class=SNPruneExperiment,
    log_timestep_freq=10,
    num_classes=NUM_CLASSES,
    batch_size=64,
    val_batch_size=128,
    extra_validations_per_epoch=1,
    validate_on_prune=True,

    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    model_class=vdrop_resnet50,
    model_args=dict(
        num_classes=NUM_CLASSES,
        z_logvar_init=-15,
        vdrop_data_class=MaskedVDropCentralData,
        conv_layer=prunable_vdrop_conv2d,
        conv_args=dict(
            target_density=conv_target_density,
        ),
        linear_layer=prunable_vdrop_linear,
        linear_args=dict(
            target_density=0.25,
        ),
    ),

    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(
        lr=6.0 / 6.0,
        momentum=0.75,
    ),

    lr_scheduler_class=None,
    lr_scheduler_args=None,

    teacher_model_class=[resnet50_swsl],

    downscale_reg_with_training_set=True,

    init_batch_norm=True,
    use_auto_augment=True,
    checkpoint_at_end=True,

    sync_to_driver=False,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
))

# 0.7465 on 32 GPUs
INITIAL_EPOCHS = 15
NUM_PRUNINGS = 6
EPOCHS_BETWEEN_PRUNINGS = 3
FINAL_EPOCHS = 30
NUM_EPOCHS = (INITIAL_EPOCHS
              + ((NUM_PRUNINGS - 1) * EPOCHS_BETWEEN_PRUNINGS)
              + FINAL_EPOCHS)
SNR_PRUNE_15_15_30 = copy.deepcopy(SNR_PRUNE_BASE)
SNR_PRUNE_15_15_30.update(dict(
    name="SNR_PRUNE_15_15_30",
    epochs_to_validate=range(NUM_EPOCHS),
    epochs=NUM_EPOCHS,

    wandb_args=dict(
        project="snr-pruning",
        name="SNR 15-15(6)-30, peak_reg=0.01",
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

    reg_scalar=make_reg_schedule(
        epochs=NUM_EPOCHS,
        pct_ramp_start=2 / 60,
        pct_ramp_end=14 / 60,
        peak_value=0.01,
        pct_drop=30 / 60,
        final_value=0.0005,
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


class ExportedExperiment(mixins.ExportModel,
                         mixins.RezeroWeights,
                         mixins.KnowledgeDistillation,
                         ImagenetExperiment):
    pass


EXPORT_TO_STATIC = copy.deepcopy(DEFAULT)
EXPORT_TO_STATIC.update(
    batch_size=128,
    experiment_class=ExportedExperiment,
    model_class=resnet50,
    model_args=dict(
        num_classes=NUM_CLASSES,
        conv_layer=sparse_conv2d,
        conv_args=dict(
            density=conv_target_density,
        ),
        linear_layer=sparse_linear,
        linear_args=dict(
            density=0.25,
        ),
    ),
)


CONFIGS = dict(
    snr_prune_15_15_30=SNR_PRUNE_15_15_30,

    snr_prune_15_15_30_exported={**EXPORT_TO_STATIC,
                                 "prev_config": SNR_PRUNE_15_15_30,
                                 "export_model_fn": maskedvdrop_to_sparseweights},
)
