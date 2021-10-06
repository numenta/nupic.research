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
import torch.nn as nn
from torchvision.datasets import STL10
from nupic.research.frameworks.greedy_infomax.utils.data_utils import get_transforms
from nupic.research.frameworks.vernon.distributed import experiments, mixins
import torch.nn.functional as F
from nupic.research.frameworks.pytorch.models.resnets import resnet50
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
    root="~/nta/data/STL10/",
    download=False,
    transform=transform_validation,
)


BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 100

class STL10SupervisedExperiment(
    experiments.SupervisedExperiment
):
    @classmethod
    def load_dataset(cls, config, train=True):
        dataset_class = config.get("dataset_class", STL10)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        dataset_args = dict(config.get("dataset_args", {}))
        if train:
            dataset_args.update(split="train")
        else:
            dataset_args.update(split="test")
        return dataset_class(**dataset_args)

class SimpleMLP(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleMLP, self).__init__()
        self.add_module("flatten", nn.Flatten())
        self.add_module("mlp", nn.Linear(in_features=in_features,
                                         out_features=out_features,
                                         bias=bias)
                        )


STL10_SUPERVISED = dict(
    experiment_class=STL10SupervisedExperiment,
    # wandb
    wandb_args=dict(
        project="stl10_supervised_experiment", name="resnet_50"
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
    val_batch_size=32,
    # Update this to stop training when accuracy reaches the metric value
    train_loader_drop_last=True,

    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),
    # Number of epochs
    epochs=NUM_EPOCHS,
    epochs_to_validate=range(0, NUM_EPOCHS, 1),
    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs
    # Model class. Must inherit from "torch.nn.Module"
    model_class=resnet50,
    # default model arguments
    model_args=dict(
        num_classes=10,
    ),
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
            max_lr=1e-3,  # change based on sparsity/dimensionality
            div_factor=5,  # initial_lr = 0.01
            final_div_factor=1000,  # min_lr = 0.0000025
            pct_start=1.0 / 10.0,
            epochs=NUM_EPOCHS,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
    ),
    loss_function=F.cross_entropy,  # each GIM layer has a cross-entropy
    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=2e-4),
    # # Learning rate scheduler class. Must inherit from "_LRScheduler"
    # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    # Noise Robustness Test Parameters
    distributed=True,
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

CONFIGS = dict(
    stl10_supervised=STL10_SUPERVISED,
)
