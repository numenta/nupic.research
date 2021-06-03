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


import unittest

import torch
from torchvision import transforms
from torchvision.datasets import FakeData

from nupic.research.frameworks.greedy_infomax.models import FullVisionModel
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    module_specific_cross_entropy,
    multiple_cross_entropy,
)
from nupic.research.frameworks.vernon import experiments

BATCH_SIZE = 32
NUM_CLASSES = 10
fake_data_args = dict(transform=transforms.ToTensor(), image_size=(1, 64, 64))

self_supervised_config = dict(
    experiment_class=experiments.SelfSupervisedExperiment,
    # Dataset
    dataset_class=FakeData,
    dataset_args=dict(
        unsupervised=fake_data_args,
        supervised=fake_data_args,
        validation=fake_data_args,
    ),
    #               STL10:
    # 500 training images (10 pre-defined folds)
    # 8000 test images (800 test images per class)
    # 100,000 unlabeled images
    num_unsupervised_samples=32,
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
    batches_in_epoch=1,
    batches_in_epoch_supervised=1,
    batches_in_epoch_val=1,
    # Update this to stop training when accuracy reaches the metric value
    # For example, stop=dict(mean_accuracy=0.75),
    stop=dict(),
    # Number of epochs
    epochs=1,
    epochs_to_validate=[],
    # Which epochs to run and report inference over the validation dataset.
    # epochs_to_validate=range(-1, 30),  # defaults to the last 3 epochs
    # Model class. Must inherit from "torch.nn.Module"
    model_class=FullVisionModel,
    # default model arguments
    model_args=dict(
        negative_samples=5,
        k_predictions=3,
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
        optimizer_args=dict(lr=1.5e-4),
    ),
    supervised_training_epochs_per_validation=1,
    loss_function=multiple_cross_entropy,  # each GIM layer has a cross-entropy
    # Optimizer class. Must inherit from "torch.optim.Optimizer"
    optimizer_class=torch.optim.Adam,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=1.5e-4),
    # # Learning rate scheduler class. Must inherit from "_LRScheduler"
    # lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
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


class SelfSupervisedLearningTest(unittest.TestCase):
    """
    This is a test class for the `SelfSupervisedExperiment` class.
    """

    def test_self_supervised_experiment(self):
        exp = self_supervised_config["experiment_class"]()
        exp.setup_experiment(self_supervised_config)
        data, target = next(iter(exp.unsupervised_loader))
        exp.encoder_optimizer.zero_grad()
        output = exp.encoder(data)
        error_loss = module_specific_cross_entropy(output, target, module=2)
        error_loss.backward()
        all_params_grads = torch.tensor(
            [
                p._grad.nonzero().sum() if p._grad is not None else 0
                for p in exp.encoder.parameters()
            ]
        )
        # Number of nonzero gradients in module 0
        module_0_nonzero_grads = all_params_grads[:15].sum()

        # Number of nonzero gradients in module 0
        module_1_nonzero_grads = all_params_grads[15:34].sum()

        # Number of nonzero gradients in module 0
        module_2_nonzero_grads = all_params_grads[34:].sum()

        self.assertEqual(module_0_nonzero_grads, 0)
        self.assertEqual(module_1_nonzero_grads, 0)
        self.assertGreater(module_2_nonzero_grads, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
