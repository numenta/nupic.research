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
This module runs a dendritic network in continual learning setting where each task
consists of learning to classify samples drawn from one of two multivariate normal
distributions.

Dendritic weights can either be hardcoded or learned. All output heads are used for
both training and inference.

Usage: adjust the config parameters `weight_init`, `dendrite_init`, and `kw` (all in
`model_args`) to control the type of forward weight initialization, dendritic weight
initialization, and k-Winners usage, respectively.
"""

import math
import pprint
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.torch.duty_cycle_metrics import max_entropy
from nupic.torch.modules import KWinners, SparseWeights, rezero_weights
from projects.dendrites.gaussian_classification.gaussian import GaussianDataset


def hardcode_dendritic_weights(dendrite_segments, context_vectors):
    """
    We hardcode the weights of dendrites such that each unit recognizes a single random
    context vector. The first dendritic segment is initialized to contain positive
    weights from that context vector. The other segment(s) ensure that the unit is
    turned off for any other context - they contain negative weights for all other
    weights.
    """
    num_units, num_segments, dim_context = dendrite_segments.weights.size()
    num_contexts, _ = context_vectors.size()
    fixed_segment_weights = torch.zeros((num_units, num_segments, dim_context))

    for i in range(num_units):
        context_perm = context_vectors[torch.randperm(num_contexts), :]
        fixed_segment_weights[i, :, :] = 1.0 * (context_perm[0, :] > 0)
        fixed_segment_weights[i, 1:, :] = -1
        fixed_segment_weights[i, 1:, :] += fixed_segment_weights[i, 0, :]
        del context_perm

    dendrite_segments.weights.data = fixed_segment_weights


# ------ Experiment class
class DendritesExperiment(mixins.RezeroWeights,
                          ContinualLearningExperiment):
    pass

    @classmethod
    def compute_task_indices(cls, config, dataset):
        # Assume dataloaders are already created
        class_indices = defaultdict(list)
        for idx, (_, _, target) in enumerate(dataset):
            class_indices[target].append(idx)

        # Defines how many classes should exist per task
        num_tasks = config.get("num_tasks", 1)
        num_classes = config.get("num_classes", None)
        assert num_classes is not None, "num_classes should be defined"
        num_classes_per_task = math.floor(num_classes / num_tasks)

        task_indices = defaultdict(list)
        for i in range(num_tasks):
            for j in range(num_classes_per_task):
                task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])
        return task_indices


# ------ Network
class DendriticMLP(nn.Module):
    """
    Dendrite segments receive context directly as input.

                    _____
                   |_____|    # classifier layer, no dendrite input
                      ^
                      |
                  _________
    context -->  |_________|  # second linear layer with dendrites
                      ^
                      |
                  _________
    context -->  |_________|  # first linear layer with dendrites
                      ^
                      |
                    input
    """

    def __init__(self, input_size, output_size, hidden_size, num_segments, dim_context,
                 weight_init, dendrite_init, kw,
                 dendritic_layer_class=AbsoluteMaxGatingDendriticLayer):

        # Forward weight initialization must of one of "kaiming" or "modified" (i.e.,
        # modified sparse Kaiming initialization)
        assert weight_init in ("kaiming", "modified")

        # Forward weight initialization must of one of "kaiming" or "modified",
        # "hardcoded"
        assert dendrite_init in ("kaiming", "modified", "hardcoded")

        super().__init__()

        self.num_segments = num_segments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dim_context = dim_context
        self.kw = kw
        self.hardcode_dendrites = (dendrite_init == "hardcoded")

        # Forward layers & k-winners
        self.dend1 = dendritic_layer_class(
            module=nn.Linear(input_size, hidden_size, bias=True),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=0.95,
            dendrite_sparsity=0.0 if self.hardcode_dendrites else 0.95,
        )
        self.dend2 = dendritic_layer_class(
            module=nn.Linear(hidden_size, hidden_size, bias=True),
            num_segments=num_segments,
            dim_context=dim_context,
            module_sparsity=0.95,
            dendrite_sparsity=0.0 if self.hardcode_dendrites else 0.95,
        )
        self.classifier = SparseWeights(module=nn.Linear(hidden_size, output_size),
                                        sparsity=0.95)

        if kw:

            print(f"Using k-Winners: 0.05 'on'")
            self.kw1 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)
            self.kw2 = KWinners(n=hidden_size, percent_on=0.05, k_inference_factor=1.0,
                                boost_strength=0.0, boost_strength_factor=0.0)

        if weight_init == "modified":

            # Scale weights to be sampled from the new inititialization U(-h, h) where
            # h = sqrt(1 / (weight_density * previous_layer_percent_on))
            init_sparse_weights(self.dend1, 0.0)
            init_sparse_weights(self.dend2, 0.95 if kw else 0.0)
            init_sparse_weights(self.classifier, 0.95 if kw else 0.0)

        if dendrite_init == "modified":
            init_sparse_dendrites(self.dend1, 0.95)
            init_sparse_dendrites(self.dend2, 0.95)

        elif dendrite_init == "hardcoded":

            # Dendritic weights will not be updated during backward pass
            for name, param in self.named_parameters():
                if "segments" in name:
                    param.requires_grad = False

    def forward(self, x, context):
        output = self.dend1(x, context=context)
        output = self.kw1(output) if self.kw else output

        output = self.dend2(output, context=context)
        output = self.kw2(output) if self.kw else output

        output = self.classifier(output)
        return output


# ------ Weight initialization functions
def init_sparse_weights(m, input_sparsity):
    """
    Modified Kaiming weight initialization that consider input sparsity and weight
    sparsity.
    """
    input_density = 1.0 - input_sparsity
    weight_density = 1.0 - m.sparsity
    _, fan_in = m.module.weight.size()
    bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
    nn.init.uniform_(m.module.weight, -bound, bound)
    m.apply(rezero_weights)


def init_sparse_dendrites(m, input_sparsity):
    """
    Modified Kaiming initialization for dendrites segments that consider input sparsity
    and dendritic weight sparsity.
    """
    # Assume `m` is an instance of `DendriticLayerBase`
    input_density = 1.0 - input_sparsity
    weight_density = 1.0 - m.segments.sparsity
    fan_in = m.segments.dim_context
    bound = 1.0 / np.sqrt(input_density * weight_density * fan_in)
    nn.init.uniform_(m.segments.weights, -bound, bound)
    m.apply(rezero_weights)


# ------ Training & evaluation function
def train_model(exp):
    # Assume `loader` yields 3-item tuples of the form (data, context, target)
    exp.model.train()
    for data, context, target in exp.train_loader:
        data = data.to(exp.device)
        context = context.to(exp.device)
        target = target.to(exp.device)

        exp.optimizer.zero_grad()
        output = exp.model(data, context)

        # Outputs are placed through a log softmax since `error_loss` is `F.nll_loss`,
        # and assumes itwill received 'logged' values
        output = F.log_softmax(output)
        error_loss = exp.error_loss(output, target)
        error_loss.backward()
        exp.optimizer.step()

        # Rezero weights if necessary
        exp.post_optimizer_step(exp.model)


def evaluate_model(exp):
    # Assume `loader` yields 3-item tuples of the form (data, context, target)
    exp.model.eval()
    total = 0

    loss = torch.tensor(0., device=exp.device)
    correct = torch.tensor(0, device=exp.device)

    with torch.no_grad():

        for data, context, target in exp.val_loader:
            data = data.to(exp.device)
            context = context.to(exp.device)
            target = target.to(exp.device)

            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_acc


if __name__ == "__main__":

    num_tasks = 50
    num_epochs = 1  # Number of training epochs per task

    config = dict(
        experiment_class=DendritesExperiment,

        dataset_class=GaussianDataset,
        dataset_args=dict(
            num_classes=2 * num_tasks,
            num_tasks=num_tasks,
            training_examples_per_class=2500,
            validation_examples_per_class=500,
            dim_x=2048,
            dim_context=2048,
        ),

        model_class=DendriticMLP,
        model_args=dict(
            input_size=2048,
            output_size=2 * num_tasks,
            hidden_size=2048,
            num_segments=2,
            dim_context=2048,
            weight_init="modified",  # Must be one of {"kaiming", "modified"}
            dendrite_init="hardcoded",  # Must be one of {"kaiming", "modified",
                                        # "hardcoded"}
            kw=False  # Turning on k-Winners results in 5% winners
        ),

        batch_size=64,
        val_batch_size=512,
        epochs=num_epochs,
        epochs_to_validate=np.arange(num_epochs),
        num_tasks=num_tasks,
        num_classes=2 * num_tasks,
        distributed=False,

        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-1),
    )

    print("Experiment config: ")
    pprint.pprint(config)
    print("")
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    if exp.model.hardcode_dendrites:

        # Manually set dendritic weights to invoke subnetworks
        hardcode_dendritic_weights(dendrite_segments=exp.model.dend1.segments,
                                   context_vectors=exp.train_loader.dataset._contexts)
        hardcode_dendritic_weights(dendrite_segments=exp.model.dend2.segments,
                                   context_vectors=exp.train_loader.dataset._contexts)

    exp.model = exp.model.to(exp.device)

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #

    # Evaluation can be slow. Only run evaluation when training task_id is in this list
    eval_tasks = [0, 3, 6, 10, 20, num_tasks - 1]
    # eval_tasks = range(num_tasks)
    for task_id in range(num_tasks):

        # Train model on current task
        t1 = time.time()
        exp.train_loader.sampler.set_active_tasks(task_id)
        for _ in range(num_epochs):
            train_model(exp)
        t2 = time.time()
        print(f"=== AFTER TASK {task_id} === TRAIN TIME: {t2 - t1}")

        # Evaluate model accuracy on each task separately
        if task_id in eval_tasks:
            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                acc_task = evaluate_model(exp)
                if isinstance(acc_task, tuple):
                    acc_task = acc_task[0]

                print(f"task {eval_task_id} accuracy: {acc_task}")

        t3 = time.time()
        print(f"=== AFTER EVALUATION, EVAL TIME: {t3 - t2}")
        print("")

    # ------------------------------------------------------------------------------- #

    # Report final aggregate accuracy
    exp.val_loader.sampler.set_active_tasks(range(num_tasks))
    acc_task = evaluate_model(exp)
    if isinstance(acc_task, tuple):
        acc_task = acc_task[0]

    print(f"Total test accuracy: {acc_task}")

    # Print entropy of layers
    max_possible_entropy = max_entropy(exp.model.hidden_size,
                                       int(0.05 * exp.model.hidden_size))
    if exp.model.kw:
        print(f"   KW1 entropy: {exp.model.kw1.entropy().item()}")
        print(f"   KW2 entropy: {exp.model.kw2.entropy().item()}")
        print(f"   max entropy: {max_possible_entropy}")
    print("")
