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


from copy import deepcopy

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets import STL10

from nupic.research.frameworks.greedy_infomax.models import FullVisionModel
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    multiple_cross_entropy,
    multiple_log_softmax_nll_loss,
    true_GIM_loss,
)
from nupic.research.frameworks.vernon.distributed import experiments, mixins
import time

class GreedyInfoMaxExperiment(
    mixins.LogEveryLoss, experiments.SelfSupervisedExperiment
):
    # def create_model(cls, config, device):
    #     model = super().create_model(config, device)
    #     use_synch_batchnorm = config.get("use_synch_batchnorm", True)
    #     distributed = config.get("distributed", False)
    #     if use_synch_batchnorm and distributed and next(model.parameters()).is_cuda:
    #         # Convert batch norm to sync batch norms
    #         model = SyncBatchNorm.convert_sync_batchnorm(module=model)
    #     return model

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


# get transforms for the dataset
def get_transforms(val=False, aug=None):
    trans = []

    if aug["randcrop"]:
        if not val:
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
transform_validation = get_transforms(val=True, aug=aug)


base_dataset_args = dict(root="~/nta/data/STL10/stl10_binary", download=False)

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
supervised_dataset_args.update(
    dict(transform=transform_supervised, split="train")
)
validation_dataset_args = deepcopy(base_dataset_args)
validation_dataset_args.update(
    dict(transform=transform_validation, split="test")
)


BATCH_SIZE = 32
NUM_CLASSES = 10
NUM_EPOCHS = 60
DEFAULT_BASE = dict(
    experiment_class=GreedyInfoMaxExperiment,
    # wandb
    wandb_args=dict(project="greedy_infomax",
                    name="paper-replication-small-true-loss"),
    # Dataset
    dataset_class=STL10,
    dataset_args=dict(
        unsupervised=unsupervised_dataset_args,
        supervised=supervised_dataset_args,
        validation=validation_dataset_args,
    ),
    #               STL10:
    # 500 training images (10 pre-defined folds)
    # 8000 test images (800 test images per class)
    # 100,000 unlabeled images
    num_unsupervised_samples=10000,
    # num_supervised_samples=500,
    # num_validation_samples=32,
    reuse_actors=True,
    # Seed
    seed=42,
    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.
    # Training batch size
    batch_size=32,
    # Supervised batch size
    batch_size_supervised=32,
    # Validation batch size
    val_batch_size=32,
    # Number of batches per epoch. Useful for debugging
    # batches_in_epoch=5,
    # batches_in_epoch_supervised=1,
    # batches_in_epoch_val=1,
    # Update this to stop training when accuracy reaches the metric value
    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),
    # Number of epochs
    epochs=NUM_EPOCHS,
    epochs_to_validate=[5, 10, 20, 30, 40, 50, 59],
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
        model_class=torch.nn.Linear,
        model_args=dict(in_features=256, out_features=NUM_CLASSES),
        loss_function=torch.nn.functional.cross_entropy,
        # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
        optimizer_class=torch.optim.Adam,
        # Optimizer class class arguments passed to the constructor
        optimizer_args=dict(lr=2e-3),
    ),
    supervised_training_epochs_per_validation=10,
    loss_function=true_GIM_loss,  # each GIM layer has a cross-entropy
    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=1.5e-3),
    # # Learning rate scheduler class. Must inherit from "_LRScheduler"
    # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
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


def load_batch_timer_func(
        model,
        loader,
        optimizer,
        device,
        criterion=None,
        complexity_loss_fn=None,
        batches_in_epoch=5,
        active_classes=None,
        pre_batch_callback=None,
        post_batch_callback=None,
        transform_to_device_fn=None,
        progress_bar=None,
):
    async_gpu = loader.pin_memory
    # time batch loading
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        t1 = time.time()
        print("Batch " + str(batch_idx) + ": " + str(t1 - t0))
        if batch_idx >= batches_in_epoch:
            break
        num_images = len(target)
        if transform_to_device_fn is None:
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)
        else:
            data, target = transform_to_device_fn(data, target, device,
                                                  non_blocking=async_gpu)
        t2 = time.time()
        print("Batch " + str(batch_idx) + " Device Transform: " + str(t2 - t1))
        t0 = t2
    print("All batches loaded")
LOAD_BATCH_TIMER = deepcopy(DEFAULT_BASE)
LOAD_BATCH_TIMER.update(dict(
    train_model_func=load_batch_timer_func,
    epochs_to_validate=[],
    batches_in_epoch=5,
    num_unsupervised_samples=10000,
    workers=0,
))

CONFIGS = dict(default_base=DEFAULT_BASE,
               load_batch_timer=LOAD_BATCH_TIMER)
