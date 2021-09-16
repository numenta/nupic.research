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

from nupic.research.frameworks.greedy_infomax.mixins.block_model_experiment import (
    BlockModelExperiment,
)
from nupic.research.frameworks.greedy_infomax.models.block_model import BlockModel
from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    MultiClassifier,
)
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    all_module_multiple_log_softmax,
    multiple_cross_entropy,
)
from nupic.research.frameworks.greedy_infomax.utils.model_utils import (
    full_resnet,
    full_sparse_resnet,
    small_resnet,
    small_sparse_resnet,
)

from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS

DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]

BATCH_SIZE = 32
NUM_EPOCHS = 10

block_wise_small_resnet_args = {"module_args": small_resnet}
block_wise_small_sparse_resnet_args = {"module_args": small_sparse_resnet}
block_wise_full_resnet_args = {"module_args": full_resnet}
block_wise_full_sparse_resnet_args = {"module_args": full_sparse_resnet}

SMALL_BLOCK = deepcopy(DEFAULT_BASE)
SMALL_BLOCK.update(
    dict(
        experiment_class=BlockModelExperiment,
        wandb_args=dict(
            project="greedy_infomax-small_block_model", name=f"base_experiment"
        ),
        epochs=NUM_EPOCHS,
        epochs_to_validate=range(NUM_EPOCHS),
        distributed=True,
        batch_size=BATCH_SIZE,
        batch_size_supervised=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        # batches_in_epoch=1,
        # batches_in_epoch_supervised=1,
        # batches_in_epoch_val=4,
        # supervised_training_epochs_per_validation=1,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        model_class=BlockModel,
        model_args=block_wise_small_resnet_args,
        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-4),
        loss_function=all_module_multiple_log_softmax,
        find_unused_parameters=True,
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
            max_lr=0.017,  # change based on sparsity/dimensionality
            div_factor=4,  # initial_lr = 0.06
            final_div_factor=200,  # min_lr = 0.0000025
            pct_start=2.0 / 10.0,
            epochs=10,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
        ),
        classifier_config=dict(
            model_class=MultiClassifier,
            model_args=dict(num_classes=10),
            loss_function=multiple_cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)

SMALL_BLOCK_LR_GRID_SEARCH = deepcopy(SMALL_BLOCK)
SMALL_BLOCK_LR_GRID_SEARCH.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-small_block_model", name=f"dense_grid_search"
        ),
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
            max_lr=tune.grid_search([0.12, 0.13, 0.14, 0.15, 0.16, 0.17]),
            # max_lr=tune.grid_search([0.19, 0.2, 0.21, 0.22, 0.213]),
            div_factor=100,  # initial_lr = 0.01
            final_div_factor=1000,  # min_lr = 0.0000025
            pct_start=1.0 / 10.0,
            epochs=10,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
        ),
    )
)


SMALL_SPARSE_BLOCK = deepcopy(SMALL_BLOCK)
SMALL_SPARSE_BLOCK.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-small_block_model", name=f"sparse_0.7_117_channels"
        ),
        model_args=block_wise_small_sparse_resnet_args,
        lr_scheduler_class=None,
        optimizer_args=dict(lr=0.16)
    )
)


SMALL_SPARSE_BLOCK_LR_GRID_SEARCH = deepcopy(SMALL_SPARSE_BLOCK)
SMALL_SPARSE_BLOCK_LR_GRID_SEARCH.update(
    wandb_args=dict(
        project="greedy_infomax-small_block_model", name=f"sparse_grid_search"
    ),
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
        max_lr=tune.grid_search([0.12, 0.14, 0.16, 0.18]),
        div_factor=100,
        final_div_factor=1000,
        pct_start=1.0 / 10.0,
        epochs=10,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
)

FULL_NUM_EPOCHS = 300
FULL_BLOCK = deepcopy(SMALL_BLOCK)
FULL_BLOCK.update(dict(
    wandb_args=dict(
        project="greedy_infomax-full_block_model",
        name=f"base_experiment"
    ),
    epochs=FULL_NUM_EPOCHS,
    epochs_to_validate=range(FULL_NUM_EPOCHS),
    model_args=block_wise_full_resnet_args,
    optimizer_class=torch.optim.SGD,
    lr_scheduler_args=dict(
        max_lr=0.017,  # change based on sparsity/dimensionality
        div_factor=4,  # initial_lr = 0.06
        final_div_factor=200,  # min_lr = 0.0000025
        pct_start=10.0 / 300.0,
        epochs=FULL_NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

FULL_SPARSE_BLOCK = deepcopy(FULL_BLOCK)
FULL_SPARSE_BLOCK.update(dict(
    wandb_args=dict(
        project="greedy_infomax-full_block_model",
        name=f"base_experiment"
    ),
    epochs=FULL_NUM_EPOCHS,
    epochs_to_validate=range(FULL_NUM_EPOCHS),
    model_args=block_wise_full_resnet_args,
    optimizer_class=torch.optim.SGD,
    lr_scheduler_args=dict(
        max_lr=0.017,  # change based on sparsity/dimensionality
        div_factor=4,  # initial_lr = 0.06
        final_div_factor=200,  # min_lr = 0.0000025
        pct_start=10.0 / 300.0,
        epochs=FULL_NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

CONFIGS = dict(
    small_block=SMALL_BLOCK,
    small_block_lr_grid_search=SMALL_BLOCK_LR_GRID_SEARCH,
    small_sparse_block=SMALL_SPARSE_BLOCK,
    small_sparse_block_lr_grid_search=SMALL_SPARSE_BLOCK_LR_GRID_SEARCH,
    full_block=FULL_BLOCK,
    full_sparse_block=FULL_SPARSE_BLOCK,
)
