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

"""
Use a prototype method for inferring context:

During training, the prototype (or "centroid") of all training samples from a
particular task are averaged in data space, and this resulting vector is used as the
context. At inference, pick the prototype closest to each test example, and use that as
context.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.torch.modules import KWinners


# ------ Experiment class
class CentroidExperiment(mixins.RezeroWeights,
                         ContinualLearningExperiment):

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Store batch size
        self.batch_size = config.get("batch_size", 1)

        # Tensor for accumulating each task's centroid vector
        self.contexts = torch.zeros((0, self.model.input_size))
        self.contexts = self.contexts.to(self.device)


# ------ Network
class CentroidDendriticMLP(DendriticMLP):
    """
    A slight variant of `DendriticMLP` which applies k-Winners to the context it
    receives.
    """

    def __init__(self, input_size, output_size, hidden_sizes, num_segments, dim_context,
                 kw, **kwargs):
        super().__init__(input_size, output_size, hidden_sizes, num_segments,
                         dim_context,
                         kw, **kwargs)

        # k-Winners module to apply to context input
        self.kw_context = KWinners(n=dim_context, percent_on=0.05,
                                   k_inference_factor=1.0, boost_strength=0.0,
                                   boost_strength_factor=1.0)

    def forward(self, x, context):
        context = self.kw_context(context)
        return super().forward(x, context)


# ------ Training & evaluation functions
def train_model(exp, context):
    exp.model.train()

    # Tile context vector
    context = context.repeat(exp.batch_size, 1)

    for data, target in exp.train_loader:
        data = data.flatten(start_dim=1)

        # Since there's only one output head, target values should be modified to be in
        # the range [0, 1, ..., 9]
        target = target % exp.num_classes_per_task

        data = data.to(exp.device)
        target = target.to(exp.device)

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
            data = data.flatten(start_dim=1)

            # Since there's only one output head, target values should be modified to
            # be in the range [0, 1, ..., 9]
            target = target % exp.num_classes_per_task

            data = data.to(exp.device)
            target = target.to(exp.device)

            # Select the closest centroid to each test example based on Euclidean
            # distance
            context = infer_centroid(exp, data)
            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_acc


def centroid(exp):
    """
    Returns the centroid vector over all training examples of `exp`'s current active
    task.
    """
    centroid_vector = torch.zeros((exp.model.input_size,))
    for batch_item in exp.train_loader:
        x = batch_item[0]
        x = x.flatten(start_dim=1)

        n_x = x.size(0)
        n = centroid_vector.size(0)

        centroid_vector = centroid_vector.to(x.device)

        centroid_vector = n * centroid_vector + x.sum(dim=0)
        centroid_vector = centroid_vector / (n + n_x)
        n += n_x

    return centroid_vector


def infer_centroid(exp, data):
    """
    Returns a 2D array where row i gives the the centroid vector closest to test
    example `data[i, :]`.
    """
    context = torch.cdist(exp.contexts, data)
    context = context.argmin(dim=0)
    context = context.cpu()
    context = exp.contexts[context]
    return context


def run_experiment(config):
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #

    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})

    for task_id in range(num_tasks):

        # Train model on current task
        exp.train_loader.sampler.set_active_tasks(task_id)

        # Construct a context vector by computing the centroid of all training examples
        context = centroid(exp).to(exp.device)
        exp.contexts = torch.cat((exp.contexts, context.unsqueeze(0)))

        for _epoch_id in range(exp.epochs):
            train_model(exp, context)

        if task_id in config["epochs_to_validate"]:

            print("")
            print(f"=== AFTER TASK {task_id} ===")
            print("")

            # Evaluate model accuracy on each task separately
            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                acc_task = evaluate_model(exp)
                if isinstance(acc_task, tuple):
                    acc_task = acc_task[0]

                print(f"task {eval_task_id} accuracy: {acc_task}")
            print("")

        else:
            print(f"--Completed task {task_id}--")

        # Reset optimizer before starting new task
        del exp.optimizer
        exp.optimizer = optimizer_class(exp.model.parameters(), **optimizer_args)

    # ------------------------------------------------------------------------------- #

    # Report final aggregate accuracy
    exp.val_loader.sampler.set_active_tasks(range(num_tasks))
    acc_task = evaluate_model(exp)
    if isinstance(acc_task, tuple):
        acc_task = acc_task[0]

    print(f"Final test accuracy: {acc_task}")
    print("")


if __name__ == "__main__":

    num_tasks = 2

    config = dict(
        experiment_class=CentroidExperiment,

        dataset_class=PermutedMNIST,
        dataset_args=dict(
            num_tasks=num_tasks,
            root=os.path.expanduser("~/nta/results/data/"),
            download=False,  # Change to True if running for the first time
            seed=np.random.randint(0, 1000)),

        model_class=CentroidDendriticMLP,
        model_args=dict(
            input_size=784,
            output_size=10,  # Single output head shared by all tasks
            hidden_sizes=[2048, 2048],
            num_segments=num_tasks,
            dim_context=784,
            kw=True,
            dendrite_weight_sparsity=0.0,
        ),

        batch_size=256,
        val_batch_size=512,
        epochs=1,
        epochs_to_validate=(4, 9, 24, 49),
        num_tasks=num_tasks,
        num_classes=10 * num_tasks,
        distributed=False,
        seed=np.random.randint(0, 10000),

        loss_function=F.cross_entropy,
        optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                           # SGD with default hyperparameter settings
        optimizer_args=dict(lr=1e-3),
    )

    run_experiment(config)
