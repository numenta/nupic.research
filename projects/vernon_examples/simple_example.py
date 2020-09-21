#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
#
# Below is a basic model/experiment config and the run function to evaluate it
#
import time

import numpy as np
import torch
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.models.common_models import StandardMLP
from nupic.research.frameworks.vernon.handlers import SupervisedExperiment

mnist_mlp = dict(
    dataset_class=datasets.MNIST,  # Using a new data-set such as datasets.CIFAR10 is
    # straightforward; simply ensure that input-size is appropriately updated
    dataset_args=dict(
        root="~/nta/datasets",
        download=True,  # Download if not present in the directory
        transform=transforms.ToTensor(),
    ),
    model_class=StandardMLP,
    model_args=dict(input_size=(28, 28), num_classes=10),
    batch_size=32,
    epochs=3,  # Number of epochs to train the network
    epochs_to_validate=np.arange(3),  # A list of the epochs to evaluate accuracy on
    num_classes=10,
    distributed=False,  # Whether or not to use Pytorch Distributed training to
    # parallelize computations across processes and clusters of machines
    experiment_class=SupervisedExperiment,  # General experiment class used to train
    # neural networks in supervised learning tasks
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(
        lr=1e-4
    ),  # Optimizer_args is used to initialize the optimizer, so any param that e.g.
    # Adam takes as an argument can be included
)


def run_experiment(config):
    exp = config.get("experiment_class")()
    exp.setup_experiment(config)
    print(f"Training started....")
    while not exp.should_stop():
        t0 = time.time()
        # print(f"Starting epoch: {exp.get_current_epoch()}")
        result = exp.run_epoch()
        print(f"Finished Epoch: {exp.get_current_epoch()}")
        print(f"Epoch Duration: {time()-t0:.1f}")
        print(f"Accuracy: {result['mean_accuracy']:.4f}")
    print(f"....Training finished")


run_experiment(mnist_mlp)
