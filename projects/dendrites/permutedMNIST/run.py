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
This module trains & evaluates a dendritic network in a continual learning setting on
permutedMNIST for a specified number of tasks/permutations. A context vector is
provided to the dendritic network, so task information need not be inferred.

This setup is very similar to that of context-dependent gating model from the paper
'Alleviating catastrophic forgetting using contextdependent gating and synaptic
stabilization' (Masse et al., 2018).
"""
import argparse
import copy

import torch
import torch.nn.functional as F

from experiments import CONFIGS
from nupic.research.frameworks.vernon.parser_utils import DEFAULT_PARSERS, process_args
from nupic.research.frameworks.vernon.run_with_raytune import run

# TODO: there are mixins that assume create_optimizer is a class method

# ------ Training & evaluation functions
def train_dendrite_model(
    model,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    complexity_loss_fn=None,
    batches_in_epoch=sys.maxsize,
    active_classes=None,
    pre_batch_callback=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
    progress_bar=None,
    share_labels=False,
):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        # TODO: need to make this more generic to not require context
        data = data[0]
        context = data[1]
        data = data.flatten(start_dim=1)

        # Since there's only one output head, target values should be modified to be in
        # the range [0, 1, ..., 9]
        if share_labels:
            target = target % exp.num_classes_per_task

        data = data.to(exp.device)
        context = context.to(exp.device)
        target = target.to(exp.device)

        optimizer.zero_grad()
        output = model(data, context)

        error_loss = criterion(output, target)
        error_loss.backward()
        optimizer.step()

        # Rezero weights if necessary
        if post_batch_callback is not None:
            post_batch_callback(model=model,
                                error_loss=error_loss.detach(),
                                complexity_loss=(complexity_loss.detach()
                                                 if complexity_loss is not None
                                                 else None),
                                batch_idx=batch_idx,
                                num_images=0,
                                time_string="")


def evaluate_model(exp):
    exp.model.eval()
    total = 0

    loss = torch.tensor(0., device=exp.device)
    correct = torch.tensor(0, device=exp.device)

    with torch.no_grad():

        for (data, context), target in exp.val_loader:
            data = data.flatten(start_dim=1)

            # Since there's only one output head, target values should be modified to
            # be in the range [0, 1, ..., 9]
            target = target % exp.num_classes_per_task

            data = data.to(exp.device)
            context = context.to(exp.device)
            target = target.to(exp.device)

            output = exp.model(data, context)

            # All output units are used to compute loss / accuracy
            loss += exp.error_loss(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    mean_acc = torch.true_divide(correct, total).item() if total > 0 else 0,
    return mean_acc


def run_experiment(config):
    exp_class = config["experiment_class"]
    exp = exp_class()
    exp.setup_experiment(config)

    # Read optimizer class and args from config as it will be used to reinitialize the
    # model's optimizer
    optimizer_class = config.get("optimizer_class", torch.optim.SGD)
    optimizer_args = config.get("optimizer_args", {})
    num_tasks = config.get("num_tasks", 2)

    # --------------------------- CONTINUAL LEARNING PHASE -------------------------- #

    for task_id in range(num_tasks):

        # Train model on current task
        exp.train_loader.sampler.set_active_tasks(task_id)
        for _ in range(exp.epochs):
            train_model(exp)

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

    parser = argparse.ArgumentParser(
        parents=DEFAULT_PARSERS,
    )
    parser.add_argument("-e", "--experiment", dest="name", default="default_base",
                        help="Experiment to run", choices=CONFIGS.keys())
    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    # Get configuration values
    config = copy.deepcopy(CONFIGS[args.name])

    # Merge configuration with command line arguments
    config.update(vars(args))

    config = process_args(args, config)
    if config is None:
        pass
    else:
        run_experiment(config)
