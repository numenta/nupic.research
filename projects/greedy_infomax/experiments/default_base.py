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

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision.datasets import STL10

from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    Classifier,
)
from nupic.research.frameworks.greedy_infomax.models.full_model import FullVisionModel
from nupic.research.frameworks.greedy_infomax.utils.data_utils import get_transforms
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    multiple_log_softmax_nll_loss,
)
from nupic.research.frameworks.vernon.distributed import experiments, mixins


class GreedyInfoMaxExperiment(
    mixins.LogEveryLoss,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    def transform_data_to_device_unsupervised(self, data, target, device, non_blocking):
        data = data.to(self.device, non_blocking=non_blocking)
        target = target.to(self.device, non_blocking=non_blocking)
        return data, target

    def create_loaders(self, config):
        unsupervised_data = self.load_dataset(config, dataset_type="unsupervised")
        if config.get("reuse_unsupervised_dataset", False):
            supervised_data = unsupervised_data
        else:
            supervised_data = self.load_dataset(config, dataset_type="supervised")
        validation_data = self.load_dataset(config, dataset_type="validation")
        num_unsupervised_samples = config.get("num_unsupervised_samples", -1)
        if num_unsupervised_samples > 0:
            unsupervised_indices = np.random.choice(
                len(unsupervised_data), num_unsupervised_samples, replace=False
            )
            unsupervised_data = Subset(unsupervised_data, unsupervised_indices)

        num_supervised_samples = config.get("num_supervised_samples", -1)
        if num_supervised_samples > 0:
            supervised_indices = np.random.choice(
                len(supervised_data), num_supervised_samples, replace=False
            )
            supervised_data = Subset(supervised_data, supervised_indices)

        num_validation_samples = config.get("num_validation_samples", -1)
        if num_validation_samples > 0:
            validation_indices = np.random.choice(
                len(validation_data), num_validation_samples, replace=False
            )
            validation_data = Subset(validation_data, validation_indices)

        self.unsupervised_loader = (
            self.train_loader
        ) = self.create_unsupervised_dataloader(config, unsupervised_data)

        self.supervised_loader = self.create_supervised_dataloader(
            config, supervised_data
        )
        self.val_loader = self.create_validation_dataloader(config, validation_data)

    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result


# labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
aug = {"randcrop": 64, "flip": True, "bw_mean": [0.4120], "bw_std": [0.2570]}
transform_unsupervised = get_transforms(val=False, aug=aug)
transform_validation = transform_supervised = get_transforms(val=True, aug=aug)

base_dataset_args = dict(root="~/nta/data/STL10/", download=False)
# base_dataset_args = dict(root="~/nta/data/STL10/stl10_binary", download=False)
unsupervised_dataset_args = deepcopy(base_dataset_args)
unsupervised_dataset_args.update(
    dict(transform=transform_unsupervised, split="unlabeled")
)
supervised_dataset_args = deepcopy(base_dataset_args)
supervised_dataset_args.update(dict(transform=transform_supervised, split="train"))
validation_dataset_args = deepcopy(base_dataset_args)
validation_dataset_args.update(dict(transform=transform_validation, split="test"))
STL10_DATASET_ARGS = dict(
    unsupervised=unsupervised_dataset_args,
    supervised=supervised_dataset_args,
    validation=validation_dataset_args,
)


BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 60
DEFAULT_BASE = dict(
    experiment_class=GreedyInfoMaxExperiment,
    # wandb
    wandb_args=dict(
        project="greedy_infomax-replication", name="s20/default-lr/full-epoch/"
    ),
    # Dataset
    dataset_class=STL10,
    dataset_args=STL10_DATASET_ARGS,
    reuse_actors=True,
    # Seed
    # seed=tune.sample_from(lambda spec: np.random.randint(1, 100)),
    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.
    # Training batch size
    batch_size=32,
    # Supervised batch size
    batch_size_supervised=32,
    # Validation batch size
    val_batch_size=32,
    # Number of batches per supervised epoch
    unsupervised_loader_drop_last=False,
    supervised_loader_drop_last=False,
    validation_loader_drop_last=False,
    # Update this to stop training when accuracy reaches the metric value
    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),
    # Number of epochs
    epochs=NUM_EPOCHS,
    epochs_to_validate=range(10, NUM_EPOCHS, 10),
    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs
    # Model class. Must inherit from "torch.nn.Module"
    model_class=FullVisionModel,
    # default model arguments
    model_args=dict(
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        grayscale=True,
        patch_size=16,
        overlap=2,
    ),
    classifier_config=dict(
        model_class=Classifier,
        model_args=dict(in_channels=256, num_classes=NUM_CLASSES),
        loss_function=torch.nn.functional.cross_entropy,
        # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
        optimizer_class=torch.optim.Adam,
        # Optimizer class class arguments passed to the constructor
        optimizer_args=dict(lr=2e-4),
    ),
    supervised_training_epochs_per_validation=30,
    reset_on_validate=False,
    loss_function=multiple_log_softmax_nll_loss,  # each GIM layer has a cross-entropy
    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=2e-4),
    # # Learning rate scheduler class. Must inherit from "_LRScheduler"
    # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    # Noise Robustness Test Parameters
    noise_levels=[0.1, 0.5, 0.9],
    # noise_mean=0,
    # noise_std=1,
    # Distributed parameters
    distributed=True,
    find_unused_parameters=True,
    # Number of dataloader workers (should be num_cpus)
    workers=0,
    local_dir="~/nta/results/greedy_infomax/experiments",
    num_samples=1,
    # How often to checkpoint (epochs)
    checkpoint_freq=0,
    keep_checkpoints_num=1,
    checkpoint_at_end=True,
    checkpoint_score_attr="training_iteration",
    # How many times to try to recover before stopping the trial
    max_failures=3,
    # How many times to retry the epoch before stopping. This is useful when
    # using distributed training with spot instances.
    max_retries=3,
    # Python Logging level : "critical", "error", "warning", "info", "debug"
    log_level="debug",
    # Python Logging Format
    log_format="%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    # Ray tune verbosity. When set to the default value of 2 it will log
    # iteration result dicts. This dict can flood the console if it contains
    # large data structures, so default to verbose=1. The SupervisedTrainable logs
    # a succinct version of the result dict.
    verbose=1,
)


SMALL_SAMPLES = deepcopy(DEFAULT_BASE)
SMALL_SAMPLES.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-replication", name="small_samples_2_epoch"
        ),
        model_args=dict(
            negative_samples=8,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
        ),
        epochs=2,
        batches_in_epoch=2,
        epochs_to_validate=[1],
    )
)


ONE_CYCLE_LR = deepcopy(DEFAULT_BASE)
ONE_CYCLE_LR.update(
    dict(
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
            max_lr=0.01,
            div_factor=50,
            final_div_factor=4000,
            pct_start=0.15,
            epochs=30,
            anneal_strategy="linear",
            max_momentum=0.01,
            cycle_momentum=False,
        ),
    )
)

VALIDATE_ONLY = deepcopy(DEFAULT_BASE)
VALIDATE_ONLY.update(
    dict(
        wandb_args=dict(project="greedy_infomax", name="validation-paper-replication"),
        epochs=61,
        epochs_to_validate=[60, 61],
        supervised_training_epochs_per_validation=20,
        num_unsupervised_samples=32,
    )
)


CONFIGS = dict(
    default_base=DEFAULT_BASE, small_samples=SMALL_SAMPLES, validate_only=VALIDATE_ONLY
)
