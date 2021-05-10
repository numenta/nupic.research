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
import pprint
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import (
    DendriticMLP,
    evaluate_dendrite_model,
    train_dendrite_model,
)
from nupic.research.frameworks.pytorch.datasets import PermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.torch.modules import KWinners


class CentroidExperiment(mixins.RezeroWeights,
                         mixins.CentroidContext,
                         mixins.PermutedMNISTTaskIndices,
                         ContinualLearningExperiment):
    pass


# ------ Network
class CentroidDendriticMLP(DendriticMLP):
    """
    A slight variant of `DendriticMLP` which applies k-Winners to the context it
    receives.
    """

    def __init__(self, input_size, output_size, hidden_sizes, num_segments, dim_context,
                 kw, kw_percent_on=0.05, context_percent_on=0.05, **kwargs):
        super().__init__(input_size, output_size, hidden_sizes, num_segments,
                         dim_context, kw, kw_percent_on, context_percent_on, **kwargs)

        # k-Winners module to apply to context input
        self.kw_context = KWinners(n=dim_context, percent_on=context_percent_on,
                                   k_inference_factor=1.0, boost_strength=0.0,
                                   boost_strength_factor=1.0)

    def forward(self, x, context):
        context = self.kw_context(context)
        return super().forward(x, context)


def run_experiment(config):
    pprint.pprint(config)
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)
    print(exp.model)

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #

    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})

    for task_id in range(num_tasks):

        # Train model on current task
        exp.train_loader.sampler.set_active_tasks(task_id)

        # Construct a context vector by computing the centroid of all training examples
        context_vector = mixins.compute_centroid(exp.train_loader).to(exp.device)
        exp.contexts = torch.cat((exp.contexts, context_vector.unsqueeze(0)))

        for _epoch_id in range(exp.epochs):
            train_dendrite_model(model=exp.model, loader=exp.train_loader,
                                 optimizer=exp.optimizer, device=exp.device,
                                 criterion=exp.error_loss, share_labels=True,
                                 num_labels=10, context_vector=context_vector,
                                 post_batch_callback=exp.post_batch_wrapper)

        if task_id in config["tasks_to_validate"]:

            print("")
            print(f"=== AFTER TASK {task_id} ===")
            print("")

            # Evaluate model accuracy on each task separately
            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                infer_centroid_fn = mixins.infer_centroid(exp.contexts)
                acc_task = evaluate_dendrite_model(model=exp.model,
                                                   loader=exp.val_loader,
                                                   device=exp.device,
                                                   criterion=exp.error_loss,
                                                   share_labels=True, num_labels=10,
                                                   infer_context_fn=infer_centroid_fn)
                acc_task = acc_task["mean_accuracy"]

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
    infer_centroid_fn = mixins.infer_centroid(exp.contexts)
    acc_task = evaluate_dendrite_model(model=exp.model, loader=exp.val_loader,
                                       device=exp.device, criterion=exp.error_loss,
                                       share_labels=True, num_labels=10,
                                       infer_context_fn=infer_centroid_fn)
    acc_task = acc_task["mean_accuracy"]

    print(f"Final test accuracy: {acc_task}")
    print("")


if __name__ == "__main__":

    num_tasks = 50

    config = dict(
        experiment_class=CentroidExperiment,

        dataset_class=PermutedMNIST,
        dataset_args=dict(
            num_tasks=num_tasks,
            root=os.path.expanduser("~/nta/results/data/"),
            download=False,  # Change to True if running for the first time
            seed=np.random.randint(0, 1000)),

        model_class=DendriticMLP,  # CentroidDendriticMLP does not affect accuracy..??
        model_args=dict(
            input_size=784,
            output_size=10,  # Single output head shared by all tasks
            hidden_sizes=[2048, 2048],
            num_segments=num_tasks,
            dim_context=784,
            kw=True,
            dendrite_weight_sparsity=0,
            weight_sparsity=0.5,
        ),

        batch_size=256,
        val_batch_size=512,
        epochs=2,
        tasks_to_validate=(0, 1, 9, 24, 49),
        num_tasks=num_tasks,
        num_classes=10 * num_tasks,
        distributed=False,
        seed=np.random.randint(0, 10000),

        loss_function=F.cross_entropy,
        optimizer_class=torch.optim.Adam,  # On permutedMNIST, Adam works better than
                                           # SGD with default hyperparameter settings
        optimizer_args=dict(lr=0.0005),
    )

    run_experiment(config)
