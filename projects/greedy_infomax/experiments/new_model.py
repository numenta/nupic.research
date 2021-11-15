# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from copy import deepcopy

import ray.tune as tune
import torch

from nupic.research.frameworks.greedy_infomax.mixins.greedy_infomax_experiment import\
    GreedyInfoMaxExperiment
from nupic.research.frameworks.greedy_infomax.models.block_model import BlockModel
from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    MultiClassifier
)
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    all_module_losses,
    multiple_cross_entropy_supervised,
)
from nupic.research.frameworks.greedy_infomax.models.resnets import ResNet7
from projects.greedy_infomax.experiments.default_base import CONFIGS

# model args for resnets
resnet_7_model_args = dict(
    channels=64,
)


DEFAULT_BASE= CONFIGS["default_base"]

# 10 epochs optimization
NUM_EPOCHS = 10
NUM_GPUS = 1

import torch.nn as nn
import torch.nn.functional as F


RESNET_7_TESTING = deepcopy(DEFAULT_BASE)
RESNET_7_TESTING.update(
    dict(
        experiment_class=GreedyInfoMaxExperiment,
        wandb_args=dict(
            project="greedy_infomax_hook_model",
            name=f"resnet_50",
        ),
        epochs=NUM_EPOCHS,
        epochs_to_validate=[
            NUM_EPOCHS - 1,
        ],
        # loss
        distributed=False,
        supervised_training_epochs_per_validation=1,
        # Uncomment this section for small batches / debugging purposes
        batches_in_epoch=2,
        batches_in_epoch_val=2,
        batches_in_epoch_supervised=2,
        batch_size=2,
        batch_size_supervised=2,
        val_batch_size=2,
        # Drop last to avoid weird batches
        unsupervised_loader_drop_last=True,
        supervised_loader_drop_last=True,
        validation_loader_drop_last=True,
        # batch_size=16 * NUM_GPUS,  # Multiply by num_gpus
        # batch_size_supervised=16 * NUM_GPUS,
        # val_batch_size=16 * NUM_GPUS,
        model_class=ResNet7,
        model_args=resnet_7_model_args,
        greedy_infomax_args=dict(
            greedy_infomax_blocks=dict(
                include_modules=[torch.nn.Conv2d],
            ),
            info_estimate_args=dict(
                k_predictions=5,
                negative_samples=16
            ),
            patchify_inputs_args=dict(
                patch_size=16,
                overlap=2,
            ),
        ),
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(lr=2e-4),
        loss_function=all_module_losses,
        find_unused_parameters=True,
        device_ids=list(range(NUM_GPUS)),
        pin_memory=False,
        lr_scheduler_class=None,
        # current best between 3e-4 and 3e-3
        cuda_launch_blocking=False,
        classifier_config=dict(
            model_class=MultiClassifier,
            model_args=dict(num_classes=10),
            loss_function=multiple_cross_entropy_supervised,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
            distributed=False,
        ),
    )
)



CONFIGS = dict(
    resnet_7_testing=RESNET_7_TESTING,
)
