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
This script runs the centroid method and saves hidden activations to later be used for
plotting.
"""

import os
import string

import numpy as np
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.torch.modules import KWinners


class CentroidContextExperiment(mixins.PlotHiddenActivations,
                                mixins.RezeroWeights,
                                mixins.PermutedMNISTTaskIndices,
                                ContinualLearningExperiment):

    def setup_experiment(self, config):
        self.batch_size = config.get("batch_size", 1)
        self.val_batch_size = config.get("val_batch_size", 1)

        super().setup_experiment(config)

        self.contexts = torch.zeros((0, self.model.input_size))
        self.contexts = self.contexts.to(self.device)


def train_model(exp, context):
    exp.model.train()
    context = context.to(exp.device)
    context = context.repeat(exp.batch_size, 1)
    for data, target in exp.train_loader:
        if isinstance(data, list):
            data, _ = data
        data = data.flatten(start_dim=1)

        if SHARE_LABELS:
            target = target % exp.num_classes_per_task

        data, target = data.to(exp.device), target.to(exp.device)

        exp.optimizer.zero_grad()
        output = exp.model(data, context)

        error_loss = exp.error_loss(output, target)
        error_loss.backward()
        exp.optimizer.step()

        # Rezero weights if necessary
        exp.post_optimizer_step(exp.model)


def evaluate_model(exp):
    exp.model.eval()
    total = 0

    loss = torch.tensor(0., device=exp.device)
    correct = torch.tensor(0, device=exp.device)

    with torch.no_grad():

        for data, target in exp.val_loader:
            if isinstance(data, list):
                data, _ = data
            data = data.flatten(start_dim=1)

            if SHARE_LABELS:
                target = target % exp.num_classes_per_task

            data, target = data.to(exp.device), target.to(exp.device)

            # Select the context by comparing distances to all context prototypes
            context = torch.cdist(exp.contexts, data)
            context = context.argmin(dim=0)
            context = context.cpu()
            context = exp.contexts[context]

            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_accuracy = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_accuracy


def tracking_eval(exp):
    """
    This function does evaluation, but modified so that

    (1) target values are converted to task values, since this function is intended to
    be called when hook tracking is on,

    (2) no values are returned.
    """
    exp.model.eval()
    with torch.no_grad():
        for data, target in exp.val_loader:
            if isinstance(data, list):
                data, _ = data
            data = data.flatten(start_dim=1)

            # This next line converts target labels to task labels, which is how we
            # want to group examples
            target = target // exp.num_classes_per_task

            data, target = data.to(exp.device), target.to(exp.device)

            # Select the context by comparing distances to all context prototypes
            context = torch.cdist(exp.contexts, data)
            context = context.argmin(dim=0)
            context = context.cpu()
            context = exp.contexts[context]

            _ = exp.model(data, context)
            if exp.ha_hook.tracking:
                # Targets were initialized on the cpu which could differ from the
                # targets collected during the forward pass.
                exp.ha_targets = exp.ha_targets.to(target.device)
                # Concatenate and discard the older targets.
                exp.ha_targets = torch.cat([target, exp.ha_targets], dim=0)
                exp.ha_targets = exp.ha_targets[:exp.ha_max_samples]


def compute_centroid(exp):
    """
    Returns the centroid vector of all samples iterated over in `loader`.
    """
    centroid_vector = torch.zeros([])
    n_centroid = 0
    for x, _ in exp.train_loader:
        if isinstance(x, list):
            x = x[0]
        x = x.flatten(start_dim=1)
        n_x = x.size(0)

        centroid_vector = centroid_vector.to(x.device)

        centroid_vector = n_centroid * centroid_vector + n_x * x.mean(dim=0)
        centroid_vector = centroid_vector / (n_centroid + n_x)
        n_centroid += n_x

    return centroid_vector


def save_hidden_activations_from_hooks():
    """
    Saves hidden activations and targets (retrieved via hooks) to disk for later
    plotting.
    """

    # A key is just a unique identifier for this run, so that previously-saved
    # files aren't overwritten
    key = "".join(np.random.choice([s for s in string.ascii_lowercase], 4))

    for name, _, hidden_activations in exp.ha_hook.get_statistics():
        targets = exp.ha_targets[:exp.ha_max_samples]

        filename = f"{NUM_TASKS}_{name}_{key}.pt"

        torch.save(hidden_activations, f"x_{filename}")
        torch.save(targets, f"y_{filename}")


if __name__ == "__main__":

    NUM_TASKS = 10
    SHARE_LABELS = True

    config = dict(
        experiment_class=CentroidContextExperiment,

        dataset_class=PermutedMNIST,
        dataset_args=dict(
            num_tasks=NUM_TASKS,
            root=os.path.expanduser("~/nta/results/data/"),
            download=False,
            seed=42
        ),

        batch_size=256,
        val_batch_size=512,
        epochs=2 if NUM_TASKS == 10 else 3 if NUM_TASKS == 50 else None,
        num_tasks=NUM_TASKS,
        num_classes=10 * NUM_TASKS,

        model_class=DendriticMLP,
        model_args=dict(
            input_size=784,
            output_size=10,  # Single output head shared by all tasks
            hidden_sizes=[2048, 2048],
            num_segments=NUM_TASKS,
            dim_context=784,
            kw=True,
            kw_percent_on=0.1,
            dendrite_weight_sparsity=0.0,
            weight_sparsity=0.5,
            context_percent_on=0.1,
        ),

        distributed=False,
        seed=np.random.randint(0, 10000),

        loss_function=F.cross_entropy,
        optimizer_class=torch.optim.Adam,
        optimizer_args=dict(lr=5e-4),

        plot_hidden_activations_args=dict(
            include_modules=[KWinners],
            plot_freq=1,
            max_samples_to_plot=5000
        ),
    )

    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #

    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})

    for task_id in range(NUM_TASKS):

        # Train model on current task
        exp.train_loader.sampler.set_active_tasks(task_id)

        # Build context vectors through centroid
        context = compute_centroid(exp).to(exp.device).detach()
        exp.contexts = torch.cat((exp.contexts, context.unsqueeze(0)))

        for _ in range(exp.epochs):
            train_model(exp, context)

        if task_id in (4, 9, 25, 49):
            print(f"--Completed task {task_id + 1}")

        # Reset optimizer before starting new task
        del exp.optimizer
        exp.optimizer = optimizer_class(exp.model.parameters(), **optimizer_args)

    # ------------------------------------------------------------------------------- #

    # Final aggregate accuracy
    exp.val_loader.sampler.set_active_tasks(range(NUM_TASKS))
    acc_task = evaluate_model(exp)
    if isinstance(acc_task, tuple):
        acc_task = acc_task[0]
    print(f"Final test accuracy: {acc_task}")

    # Iterate over all test examples with hook tracking (used for plotting)
    exp.ha_hook.start_tracking()
    tracking_eval(exp)
    exp.ha_hook.stop_tracking()

    save_hidden_activations_from_hooks()
