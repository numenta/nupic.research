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
from nupic.research.frameworks.dendrites import evaluate_dendrite_model, train_dendrite_model
from nupic.research.frameworks.vernon.parser_utils import DEFAULT_PARSERS, process_args
from nupic.research.frameworks.vernon.run_with_raytune import run

# TODO: there are mixins that assume create_optimizer is a class method

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
            train_dendrite_model(model=exp.model, loader=exp.train_loader,
                                 optimizer=exp.optimizer, device=exp.device,
                                 criterion=exp.error_loss, share_labels=True,
                                 num_labels=10, post_batch_callback=exp.post_batch_wrapper)

        if task_id in config["epochs_to_validate"]:

            print("")
            print(f"=== AFTER TASK {task_id} ===")
            print("")

            # Evaluate model accuracy on each task separately
            for eval_task_id in range(task_id + 1):

                exp.val_loader.sampler.set_active_tasks(eval_task_id)
                results = evaluate_dendrite_model(model=exp.model, loader=exp.val_loader,
                                                  device=exp.device,
                                                  criterion=exp.error_loss,
                                                  share_labels=True, num_labels=10)
                acc_task = results["mean_accuracy"]
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
    results = evaluate_dendrite_model(model=exp.model, loader=exp.val_loader,
                                      device=exp.device, criterion=exp.error_loss,
                                      share_labels=True, num_labels=10)
    acc_task = results["mean_accuracy"]
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
