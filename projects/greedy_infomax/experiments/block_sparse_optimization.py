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

from nupic.research.frameworks.greedy_infomax.mixins.data_parallel_block_model_experiment import DataParallelBlockModelExperiment # noqa E501
from nupic.research.frameworks.greedy_infomax.models.block_model import BlockModel
from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    MultiClassifier,
)
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    all_module_losses,
    multiple_cross_entropy_supervised,
)
from nupic.research.frameworks.greedy_infomax.utils.model_utils import (
    small_resnet,
    small_sparse_70_resnet,
    small_sparse_80_resnet,
)
from projects.greedy_infomax.experiments.block_wise_training import (
    CONFIGS as BLOCK_WISE_CONFIGS,
)

FULL_RESNET_50 = BLOCK_WISE_CONFIGS["full_resnet_50"]

# 5 epochs SigOpt optimization
NUM_EPOCHS = 10
NUM_GPUS = 8

small_dense_resnet_args = {"module_args": small_resnet}
small_sparse_70_resnet_args = {"module_args": small_sparse_70_resnet}
small_sparse_80_resnet_args = {"module_args": small_sparse_80_resnet}

SMALL_DENSE_ONE_CYCLE_GRID_SEARCH = deepcopy(FULL_RESNET_50)
SMALL_DENSE_ONE_CYCLE_GRID_SEARCH.update(dict(
    experiment_class=DataParallelBlockModelExperiment,
    wandb_args=dict(
        project="greedy_infomax_small_resnet_onecycle",
        name=f"small_dense"
    ),
    epochs=NUM_EPOCHS,
    epochs_to_validate=[NUM_EPOCHS - 1, ],
    # loss
    distributed=False,
    supervised_training_epochs_per_validation=50,
    # Uncomment this section for small batches / debugging purposes
    # batches_in_epoch=2,
    # batches_in_epoch_val=2,
    # batches_in_epoch_supervised=2,
    # batch_size = 2,
    # batch_size_supervised=2,
    # val_batch_size=2,

    # Drop last to avoid weird batches
    unsupervised_loader_drop_last=True,
    supervised_loader_drop_last=True,
    validation_loader_drop_last=True,

    batch_size=16 * NUM_GPUS,  # Multiply by num_gpus
    batch_size_supervised=16 * NUM_GPUS,
    val_batch_size=16 * NUM_GPUS,
    model_class=BlockModel,
    model_args=small_dense_resnet_args,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=2e-4),
    loss_function=all_module_losses,
    find_unused_parameters=True,
    device_ids=list(range(NUM_GPUS)),
    pin_memory=False,
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
        # 0.001, 0.003, 0.01
        # 0.006, 0.008
        max_lr=tune.grid_search([0.0045, 0.0055, 0.0065]),
        # max_lr=tune.grid_search([0.19, 0.2, 0.21, 0.22, 0.213]),
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=1.0 / 10.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
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
))

SMALL_SPARSE_70_ONE_CYCE_GRID_SEARCH = deepcopy(SMALL_DENSE_ONE_CYCLE_GRID_SEARCH)
SMALL_SPARSE_70_ONE_CYCE_GRID_SEARCH.update(
    wandb_args=dict(
        project="greedy_infomax_small_resnet_onecycle",
        name=f"small_sparse_70"
    ),
    model_args=small_sparse_70_resnet_args,
    lr_scheduler_args=dict(
        max_lr=tune.grid_search([0.0045, 0.0055, 0.006]),
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=1.0 / 10.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
)


SMALL_SPARSE_80_ONE_CYCE_GRID_SEARCH = deepcopy(SMALL_DENSE_ONE_CYCLE_GRID_SEARCH)
SMALL_SPARSE_80_ONE_CYCE_GRID_SEARCH.update(
    wandb_args=dict(
        project="greedy_infomax_small_resnet_onecycle",
        name=f"small_sparse_80"
    ),
    model_args=small_sparse_80_resnet_args,
    lr_scheduler_args=dict(
        max_lr=tune.grid_search([0.004, 0.008, 0.012]),
        # max_lr=tune.grid_search([0.19, 0.2, 0.21, 0.22, 0.213]),
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=1.0 / 10.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
)

CONFIGS = dict(
    small_dense_one_cycle=SMALL_DENSE_ONE_CYCLE_GRID_SEARCH,
    small_sparse_70_one_cycle=SMALL_SPARSE_70_ONE_CYCE_GRID_SEARCH,
    small_sparse_80_one_cycle=SMALL_SPARSE_80_ONE_CYCE_GRID_SEARCH,
)
