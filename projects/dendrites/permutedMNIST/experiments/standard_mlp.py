#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Base Experiment configuration.
"""

import os

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.models import StandardMLP
from nupic.research.frameworks.vernon import SupervisedExperiment

"""Regular MNIST """


NUM_TASKS = 2

# Gets about 98% accuracy
DENSE_MLP = dict(
    experiment_class=SupervisedExperiment,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/dendrites"),

    dataset_class=datasets.MNIST,
    dataset_args=dict(
        # Consistent location outside of git repo
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,  # Change to True if running for the first time
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13062755,), (0.30810780,)),
        ])
    ),

    model_class=StandardMLP,
    model_args=dict(
        input_size=784,
        num_classes=10,
        hidden_sizes=[2048, 2048],
    ),

    batch_size=32,
    val_batch_size=512,
    epochs=13,
    epochs_to_validate=range(30),
    num_classes=10,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    num_samples=1,  # Increase to run multiple experiments in parallel

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(
        lr=0.01,
    ),

    # Learning rate scheduler class and args. Must inherit from "_LRScheduler"
    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_args=dict(
        gamma=0.1,
        step_size=10,
    ),
)

# Export configurations in this file
CONFIGS = dict(
    dense_mlp=DENSE_MLP,
)
