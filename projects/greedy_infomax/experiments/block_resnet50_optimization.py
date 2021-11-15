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

from nupic.research.frameworks.greedy_infomax.mixins.data_parallel_block_model_experiment import (  # noqa E501
    DataParallelBlockModelExperiment,
)
from nupic.research.frameworks.greedy_infomax.models.block_model import BlockModel
from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    MultiClassifier,
)
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    all_module_losses,
    multiple_cross_entropy_supervised,
)
from nupic.research.frameworks.greedy_infomax.utils.model_utils import \
    full_resnet_50, full_resnet_50_sparse_70, full_resnet_50_sparse_80
from projects.greedy_infomax.experiments.block_wise_training import (
    CONFIGS as BLOCK_WISE_CONFIGS,
)

FULL_RESNET_50 = BLOCK_WISE_CONFIGS["full_resnet_50"]

# 10 epochs optimization
NUM_EPOCHS = 10
NUM_GPUS = 8

block_wise_full_resnet_50_args = {"module_args": full_resnet_50}
block_wise_full_resnet_50_sparse_70_args = {"module_args": full_resnet_50_sparse_70}
block_wise_full_resnet_50_sparse_80_args = {"module_args": full_resnet_50_sparse_80}

RESNET_50_ONE_CYCLE_LR_GRID_SEARCH = deepcopy(FULL_RESNET_50)
RESNET_50_ONE_CYCLE_LR_GRID_SEARCH.update(
    dict(
        experiment_class=DataParallelBlockModelExperiment,
        wandb_args=dict(
            project="greedy_infomax_full_resnet_onecycle",
            name=f"onecycle_grid_search_iteration_3",
        ),
        epochs=NUM_EPOCHS,
        epochs_to_validate=[
            NUM_EPOCHS - 1,
        ],
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
        model_args=block_wise_full_resnet_50_args,
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(lr=2e-4),
        loss_function=all_module_losses,
        find_unused_parameters=True,
        device_ids=list(range(NUM_GPUS)),
        pin_memory=False,
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        # current best between 3e-4 and 3e-3
        lr_scheduler_args=dict(
            max_lr=tune.grid_search([0.0013, 0.0015, 0.0017]),
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
    )
)

RESNET_50_SPARSE_70_ONE_CYCLE_LR_GRID_SEARCH= deepcopy(
    RESNET_50_ONE_CYCLE_LR_GRID_SEARCH)
RESNET_50_SPARSE_70_ONE_CYCLE_LR_GRID_SEARCH.update(dict(
    wandb_args=dict(
        project="greedy_infomax_full_resnet_onecycle",
        name=f"onecycle_grid_search_sparse_70",
    ),
    epochs=10,
    model_args=block_wise_full_resnet_50_sparse_70_args,
    lr_scheduler_args=dict(
        max_lr=tune.grid_search([0.0001, 0.0003, 0.001, 0.003]),
        # max_lr=tune.grid_search([0.19, 0.2, 0.21, 0.22, 0.213]),
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=1.0 / 10.0,
        epochs=10,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

RESNET_50_SPARSE_80_ONE_CYCLE_LR_GRID_SEARCH= deepcopy(
    RESNET_50_ONE_CYCLE_LR_GRID_SEARCH)
RESNET_50_SPARSE_80_ONE_CYCLE_LR_GRID_SEARCH.update(dict(
    wandb_args=dict(
        project="greedy_infomax_full_resnet_onecycle",
        name=f"onecycle_grid_search_sparse_80",
    ),
    epochs=10,
    model_args=block_wise_full_resnet_50_sparse_80_args,
    lr_scheduler_args=dict(
        max_lr=tune.grid_search([0.0001, 0.0003, 0.001, 0.003]),
        # max_lr=tune.grid_search([0.19, 0.2, 0.21, 0.22, 0.213]),
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=1.0 / 10.0,
        epochs=10,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

NUM_EPOCHS = 100
RESNET_50_ONE_CYCLE_LR_FULL = deepcopy(RESNET_50_ONE_CYCLE_LR_GRID_SEARCH)
RESNET_50_ONE_CYCLE_LR_FULL.update(dict(
    wandb_args=dict(
            project="greedy_infomax_full_resnet_onecycle",
            name=f"onecycle_full",
    ),
    epochs=NUM_EPOCHS,
    epochs_to_validate=[50, 100, 150, 200, 290, 295, 299],
    supervised_training_epochs_per_validation=100,
    lr_scheduler_args=dict(
        max_lr=0.0013, #found through grid search
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=3.0 / 10.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

NUM_EPOCHS=300
RESNET_50_DENSE_ONE_CYCLE_300 = deepcopy(RESNET_50_ONE_CYCLE_LR_FULL)
RESNET_50_DENSE_ONE_CYCLE_300.update(dict(
    wandb_args=dict(
            project="greedy_infomax_full_resnet_300",
            name=f"dense",
    ),
    epochs=NUM_EPOCHS,
    epochs_to_validate=[10, 25, 50, 100, 150, 200, 290, 295, 299],
    supervised_training_epochs_per_validation=50,
    lr_scheduler_args=dict(
        max_lr=0.0013, #found through grid search
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=2.0 / 300.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

import pickle
import io
from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
)
from nupic.research.frameworks.vernon.network_utils import (
    get_compatible_state_dict,
)
class LoadBlockModelExperiment(DataParallelBlockModelExperiment):
    def setup_experiment(self, config):
        super(LoadBlockModelExperiment, self).setup_experiment(config)
        self.load_file = config.get("model_path", None)
        if self.load_file is not None:
            with open(self.load_file, mode="rb") as f:
                state = pickle.load(f)
            if "model" in state:
                with io.BytesIO(state["model"]) as buffer:
                    state_dict = deserialize_state_dict(buffer, self.device)
                model = self.model
                if hasattr(model, "module"):
                    # DistributedDataParallel
                    model = model.module
                state_dict = get_compatible_state_dict(state_dict, model)
                model.load_state_dict(state_dict)

RESNET_50_DENSE_TWO_CYCLE_300 = deepcopy(RESNET_50_DENSE_ONE_CYCLE_300)
RESNET_50_DENSE_TWO_CYCLE_300.update(dict(
    experiment_class=DataParallelBlockModelExperiment,
    wandb_args=dict(
            project="greedy_infomax_full_resnet_300",
            name=f"dense_two_cycle",
    ),
    epochs=30,
    epochs_to_validate=[0, 5, 10, 15, 20, 25, 29],
    checkpoint_file="/home/ec2-user/nta/results/greedy_infomax/experiments"
               "/resnet_50_dense_one_cycle_300/RemoteProcessTrainable_0_2021-10-28_18"
               "-07-034z8mwnuw/checkpoint_300/checkpoint",
    lr_scheduler_args=dict(
        max_lr=0.0001,
        div_factor=10,
        final_div_factor=100,
        pct_start=3.0 / 30.0,
        epochs=30,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

RESNET_50_SPARSE_70_ONE_CYCLE_300 = deepcopy(RESNET_50_DENSE_ONE_CYCLE_300)
RESNET_50_SPARSE_70_ONE_CYCLE_300.update(dict(
    wandb_args=dict(
            project="greedy_infomax_full_resnet_300",
            name=f"sparse_70",
    ),
    model_args=block_wise_full_resnet_50_sparse_70_args,
    lr_scheduler_args=dict(
        max_lr=0.001, #found through grid search
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=2.0 / 300.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))

RESNET_50_SPARSE_70_TWO_CYCLE_300 = deepcopy(RESNET_50_SPARSE_70_ONE_CYCLE_300)
RESNET_50_SPARSE_70_TWO_CYCLE_300.update(dict(
    experiment_class=LoadBlockModelExperiment,
    wandb_args=dict(
            project="greedy_infomax_full_resnet_300",
            name=f"sparse_70_two_cycle",
    ),
    epochs=30,
    epochs_to_validate=[0, 5, 10, 15, 20, 25, 29],
    model_path="/home/ec2-user/nta/results/greedy_infomax/experiments"
               "/resnet_50_sparse_70_one_cycle_300/RemoteProcessTrainable_0_2021-10"
               "-29_23-52-397grdrlb6/checkpoint_300/checkpoint",
    lr_scheduler_args=dict(
        max_lr=0.0001,
        div_factor=10,
        final_div_factor=100,
        pct_start=3.0 / 30.0,
        epochs=30,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
    ))


RESNET_50_SPARSE_70_ONE_CYCLE_100 = deepcopy(RESNET_50_SPARSE_70_ONE_CYCLE_300)
RESNET_50_SPARSE_70_ONE_CYCLE_100.update(dict(
    wandb_args=dict(
            project="greedy_infomax_full_resnet_100",
            name=f"sparse_70",
    ),
    epochs=100,
    lr_scheduler_args=dict(
        max_lr=0.001, #found through grid search
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=3.0 / 100.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
))




RESNET_50_SPARSE_80_ONE_CYCLE_300= deepcopy(RESNET_50_SPARSE_70_ONE_CYCLE_300)
RESNET_50_SPARSE_80_ONE_CYCLE_300.update(dict(
    wandb_args=dict(
            project="greedy_infomax_full_resnet_300",
            name=f"sparse_80",
    ),
    model_args=block_wise_full_resnet_50_sparse_80_args,
    epochs=NUM_EPOCHS,
    epochs_to_validate=[100, 200, 290, 295, 299],
    supervised_training_epochs_per_validation=100,
    lr_scheduler_args=dict(
        max_lr=0.0013, #found through grid search
        div_factor=100,  # initial_lr = 0.01
        final_div_factor=1000,  # min_lr = 0.0000025
        pct_start=2.0 / 300.0,
        epochs=NUM_EPOCHS,
        anneal_strategy="linear",
        max_momentum=1e-4,
        cycle_momentum=False,
    ),
    ))




CONFIGS = dict(
    resnet_50_one_cycle_lr_grid_search=RESNET_50_ONE_CYCLE_LR_GRID_SEARCH,
    resnet_50_sparse_70_one_cycle_lr_grid_search=RESNET_50_SPARSE_70_ONE_CYCLE_LR_GRID_SEARCH,
    resnet_50_sparse_80_one_cycle_lr_grid_search=RESNET_50_SPARSE_80_ONE_CYCLE_LR_GRID_SEARCH,
    resnet_50_one_cycle_lr_full=RESNET_50_ONE_CYCLE_LR_FULL,
    resnet_50_dense_one_cycle_300=RESNET_50_DENSE_ONE_CYCLE_300,
    resnet_50_dense_two_cycle_300=RESNET_50_DENSE_TWO_CYCLE_300,
    resnet_50_sparse_70_one_cycle_300=RESNET_50_SPARSE_70_ONE_CYCLE_300,
    resnet_50_sparse_70_two_cycle_300=RESNET_50_SPARSE_70_TWO_CYCLE_300,
    resnet_50_sparse_70_one_cycle_100=RESNET_50_SPARSE_70_ONE_CYCLE_100,
    resnet_50_sparse_80_one_cycle_300=RESNET_50_SPARSE_80_ONE_CYCLE_300,
)
