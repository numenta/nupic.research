#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#


import sys
from copy import deepcopy

import numpy as np
import torch
from ray import tune
from torchvision import transforms
from torchvision.datasets import STL10

from nupic.research.frameworks.greedy_infomax.models import FullVisionModel
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    multiple_cross_entropy,
)
from nupic.research.frameworks.vernon import experiments, mixins


class GreedyInfoMaxExperiment(
    mixins.LogEveryLoss,
    experiments.SelfSupervisedExperiment,
):
    pass


# get transforms for the dataset
def get_transforms(val=False, aug=None):
    trans = []

    if aug["randcrop"]:
        if val:
            trans.append(transforms.RandomCrop(aug["randcrop"]))
        else:
            trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not val:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans


# labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
aug = {
    "randcrop": 64,
    "flip": True,
    "grayscale": True,
    "mean": [0.4313, 0.4156, 0.3663],
    "std": [0.2683, 0.2610, 0.2687],
    "bw_mean": [0.4120],
    "bw_std": [0.2570],
}
transform_unsupervised = transform_supervised = get_transforms(val=False, aug=aug)
transform_validation = trans = get_transforms(val=True, aug=aug)


base_dataset_args = dict(root="~/nta/data/STL10", download=False)

# #fake data class for debugging purposes
# def fake_data(size=256, image_size=(3, 96, 96), num_classes = 10, train=True,
#               transform=transform_validation):
#     return FakeData(size=size, image_size=image_size, num_classes=num_classes,
#     transform=transform)

unsupervised_dataset_args = deepcopy(base_dataset_args)
unsupervised_dataset_args.update(
    dict(transform=transform_unsupervised, split="unlabeled")
)
supervised_dataset_args = deepcopy(base_dataset_args)
supervised_dataset_args.update(dict(transform=transform_supervised, split="train"))
validation_dataset_args = deepcopy(base_dataset_args)
validation_dataset_args.update(dict(transform=transform_validation, split="test"))


BATCH_SIZE = 32
NUM_CLASSES = 10
DEFAULT_BASE = dict(
    experiment_class=GreedyInfoMaxExperiment,
    # Dataset
    dataset_class=STL10,
    dataset_args=dict(
        unsupervised=unsupervised_dataset_args,
        supervised=supervised_dataset_args,
        validation=validation_dataset_args,
    ),
    num_unsupervised_samples=32,
    num_supervised_samples=32,
    num_validation_samples=32,
    reuse_actors=True,
    # Seed
    seed=tune.sample_from(lambda spec: np.random.randint(1, 10000)),
    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.
    # Training batch size
    batch_size=BATCH_SIZE,
    # Validation batch size
    val_batch_size=10000,
    # Number of batches per epoch. Useful for debugging
    batches_in_epoch=sys.maxsize,
    # Update this to stop training when accuracy reaches the metric value
    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),
    # Number of epochs
    epochs=10,
    epochs_to_validate=[-1],
    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs
    # Model class. Must inherit from "torch.nn.Module"
    model_class=FullVisionModel,
    # default model arguments
    model_args=dict(
        negative_samples=10,
        k_predictions=5,
        resnet_50=False,
        grayscale=True,
        patch_size=16,
        overlap=2,
    ),
    classifier_config=dict(
        model_class=torch.nn.Linear,
        model_args=dict(in_features=100, out_features=NUM_CLASSES),
        loss_function=torch.nn.functional.cross_entropy,
        # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
        optimizer_class=torch.optim.Adam,
        # Optimizer class class arguments passed to the constructor
        optimizer_args=dict(lr=1.5e-4),
    ),
    loss_function=multiple_cross_entropy,  # each GIM layer has a cross-entropy
    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=1.5e-4),
    # # Learning rate scheduler class. Must inherit from "_LRScheduler"
    # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
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


CONFIGS = dict(default_base=DEFAULT_BASE)
