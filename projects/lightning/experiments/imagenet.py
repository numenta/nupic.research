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

import os

import torch

from nupic.research.frameworks.lightning import mixins
from nupic.research.frameworks.lightning.models import SupervisedModel
from nupic.research.frameworks.pytorch import datasets
from nupic.research.frameworks.pytorch.models.pretrained_models import (
    resnet50_swsl,
    resnext50_32x4d_swsl,
    se_resnet50,
    se_resnext50_32x4d,
)
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.research.frameworks.pytorch.modules import (
    relu_maybe_kwinners2d,
    sparse_conv2d,
    sparse_linear,
)


def conv_weight_density(in_channels, out_channels, kernel_size):
    weights_per_channel = kernel_size * kernel_size * in_channels
    if weights_per_channel < 100:
        density = 0.7

    elif weights_per_channel < 200:
        density = 0.5

    elif weights_per_channel < 500:
        density = 0.4

    elif weights_per_channel < 1000:
        density = 0.3

    elif weights_per_channel < 2000:
        density = 0.2

    elif weights_per_channel < 4000:
        density = 0.2

    else:
        density = 0.15

    return density


def conv_activation_density(out_channels):
    percent_on = 1.0
    if out_channels >= 128:
        percent_on = 0.3

    return percent_on


class SparseSupervisedModelKD(mixins.RezeroWeights,
                              mixins.UpdateBoostStrength,
                              mixins.KnowledgeDistillation,
                              SupervisedModel):
    pass


NUM_CLASSES = 1000
EPOCHS = 120

SPARSE1000_R1 = dict(
    lightning_model_class=SparseSupervisedModelKD,
    lightning_model_args=dict(
        config=dict(
            batch_norm_weight_decay=False,
            init_batch_norm=True,

            epochs=EPOCHS,

            num_classes=NUM_CLASSES,
            batch_size=128,

            teacher_model_class=[resnext50_32x4d_swsl, resnet50_swsl,
                                 se_resnext50_32x4d, se_resnet50],
            kd_ensemble_weights=[0.3, 0.3, 0.2, 0.2],
            kd_factor_init=1,

            dataset_class=datasets.imagenet,
            dataset_args=dict(
                data_path=os.path.expanduser("~/nta/data/imagenet/imagenet.hdf5"),
                num_classes=NUM_CLASSES,
                use_auto_augment=True,
            ),
            workers=4,

            model_class=resnet50,
            model_args=dict(
                num_classes=NUM_CLASSES,
                conv_layer=sparse_conv2d,
                conv_args=dict(density=conv_weight_density),
                linear_layer=sparse_linear,
                linear_args=dict(density=0.25),
                act_layer=relu_maybe_kwinners2d,
                act_args=dict(density=conv_activation_density),
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
            lr_scheduler_args=dict(
                max_lr=6.0,
                div_factor=6,  # initial_lr = 1.0
                final_div_factor=4000,  # min_lr = 0.00025
                pct_start=5.0 / 120.0,
                epochs=120,
                anneal_strategy="linear",
                max_momentum=0.01,
                cycle_momentum=False,
            ),

            optimizer_args=dict(
                lr=0.1,
                weight_decay=0.00005,
                momentum=0.0,
                nesterov=False,
            ),
        ),
    ),

    lightning_trainer_args=dict(
        # No early stopping
        min_epochs=EPOCHS,
        max_epochs=EPOCHS,
    ),
)


CONFIGS = dict(
    sparse_1000_r1_kd=SPARSE1000_R1,
)
