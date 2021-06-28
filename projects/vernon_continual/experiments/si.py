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

"""
Experiment file that runs Synaptic Intelligence (Zenke, Poole, Ganguli (2017)) in a
continual learning experiment on permutedMNIST.
"""

import os
from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn.functional as F
from torch import nn

from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins


class SIContinualLearningExperiment(mixins.SynapticIntelligence,
                                    mixins.PermutedMNISTTaskIndices,
                                    ContinualLearningExperiment):
    pass


def train_si_model(model, loader, optimizer, device, criterion=F.cross_entropy,
                   post_batch_callback=None, **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.flatten(start_dim=1)

        # Share labels between tasks
        # TODO don't hardcode num classes per task
        target = target % 10

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        error_loss = criterion(output, target)
        error_loss.backward()
        optimizer.step()

        # Rezero weights if necessary
        if post_batch_callback is not None:
            post_batch_callback(model=model, error_loss=error_loss.detach(),
                                complexity_loss=None, batch_idx=batch_idx,
                                num_images=0, time_string="")


def evaluate_si_model(model, loader, device, criterion=F.cross_entropy, **kwargs):
    model.eval()
    total = 0

    loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)

    with torch.no_grad():
        for data, target in loader:
            data = data.flatten(start_dim=1)

            # Share labels between tasks
            # TODO don't hardcode num classes per task
            target = target % 10

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    results = {
        "total_correct": correct.item(),
        "total_tested": total,
        "mean_loss": loss.item() / total if total > 0 else 0,
        "mean_accuracy": torch.true_divide(correct, total).item() if total > 0 else 0,
    }
    return results


class SINetwork(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.classifier = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = self.classifier(output)
        return output


NUM_TASKS = 5

# Synaptic Intelligence on permutedMNIST with 5 tasks
SI_PERMUTEDMNIST_5 = dict(
    experiment_class=SIContinualLearningExperiment,
    num_samples=8,

    # Results path
    local_dir=os.path.expanduser("~/nta/results/experiments/si"),

    dataset_class=PermutedMNIST,
    dataset_args=dict(
        num_tasks=5,
        root=os.path.expanduser("~/nta/results/data/"),
        download=False,
        seed=42,
    ),

    model_class=SINetwork,
    model_args=dict(
        input_size=784,
        hidden_sizes=[2000, 2000],
        output_size=10,
    ),

    si_args=dict(
        c=0.1,
        damping=0.1,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=20,
    tasks_to_validate=(0, 1, 4, 9),
    num_tasks=5,
    num_classes=10 * 5,
    distributed=False,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

    loss_function=F.cross_entropy,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=1e-3),
    reset_optimizer_after_task=False,  # The authors report not resetting the Adam
                                       # optimizer as this works better in practice

    train_model_func=train_si_model,
    evaluate_model_func=evaluate_si_model,
)

# Synaptic Intelligence on permutedMNIST with 10 tasks
SI_PERMUTEDMNIST_10 = deepcopy(SI_PERMUTEDMNIST_5)
SI_PERMUTEDMNIST_10["dataset_args"].update(num_tasks=10)
SI_PERMUTEDMNIST_10.update(
    num_tasks=10,
    num_classes=10 * 10,
)

CONFIGS = dict(
    si_permutedmnist_5=SI_PERMUTEDMNIST_5,
    si_permutedmnist_10=SI_PERMUTEDMNIST_10,
)
