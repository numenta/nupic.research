# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import ray
import torch
from ray import tune

from nupic.research.frameworks.backprop_structure.ray_ax.ray_ax_utils import (
    AxSearch,
    ax_client_with_explicit_strategy,
    get_ray_trials,
)
from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)


def hyperparameter_loss(config, result):
    log_error = np.log(np.maximum(1 - result["mean_accuracy"], 1e-7))
    return log_error


def get_best_config(experiment_dir, parameters, num_training_iterations):
    known_trials = get_ray_trials(experiment_dir, parameters)
    known_trials = [
        (config, result)
        for config, result in known_trials
        if result["training_iteration"] == num_training_iterations]

    grouped_trials = defaultdict(list)
    for config, result in known_trials:
        grouped_trials[config].append(result)
    grouped_trials = list(grouped_trials.items())

    max_trial = None
    max_acc = -1
    for trial in grouped_trials:
        config, results = trial
        acc = np.mean([result["mean_accuracy"] for result in results])
        if acc > max_acc:
            max_trial = trial
            max_acc = acc

    return max_trial


def insert_hyperparameter_loss(trainable):
    """
    Create a new Ray Trainable class that inserts the hyperparameter loss into
    each result.
    """

    class Cls(trainable):
        def _setup(self, config):
            self.config = config
            super()._setup(config)

        def _train(self):
            result = super()._train()
            result["hyperparameter_loss"] = hyperparameter_loss(self.config,
                                                                result)
            return result

    Cls.__name__ = trainable.__name__

    return Cls


def ax_optimize_accuracy(exploring_trainable, followup_trainable,
                         experiment_name, script_dir, parameters,
                         num_training_iterations, num_best_config_samples):
    """
    Optimize a Ray Trainable's "mean_accuracy", using Ax to select
    hyperparameters. This method will pick up any existing results for this
    experiment and plug them into the Ax model, then will generate new results.
    Finally, this procedure finds the best configuration and collects additional
    results for the configuration to ensure that it is well-sampled.
    """

    experiment_dir = os.path.expanduser(f"~/ray_results/{experiment_name}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-initial-configs", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--num-random", type=int, default=16,
                        help="The number of random configs to test.")
    parser.add_argument("--num-configs", type=int, default=16,
                        help="The number of non-random configs to test.")
    args = parser.parse_args()

    ray.init()

    if args.run_initial_configs:
        input_dir = Path(script_dir) / "best_configs"
        input_file = input_dir / f"{experiment_name}.json"
        if input_file.exists():
            print("STAGE 0: Test initial suggested hyperparameters")
            with open(input_file, "r") as f:
                best_trial = json.load(f)

            best_config, _ = best_trial
            best_config = dict(best_config)
            # Only include parameters specified in the parameters list.
            best_config = {parameter["name"]: best_config[parameter["name"]]
                           for parameter in parameters}

            tune.run(
                exploring_trainable,
                name=experiment_name,
                config=dict(best_config),
                num_samples=1,
                resources_per_trial={
                    "cpu": 1,
                    "gpu": (1 if torch.cuda.is_available() else 0)},
                loggers=DEFAULT_LOGGERS,
                verbose=1,
            )

    num_random_remaining = args.num_random

    for _ in range(args.num_iterations):
        if num_random_remaining + args.num_configs > 0:
            print("STAGE 1: Hyperparameter search")
            known_trials = get_ray_trials(experiment_dir, parameters)
            known_trials = [
                (config, result)
                for config, result in known_trials
                if result["training_iteration"] == num_training_iterations]

            ax_client = ax_client_with_explicit_strategy(num_random_remaining,
                                                         args.num_configs)
            ax_client.create_experiment(parameters=parameters,
                                        objective_name="hyperparameter_loss",
                                        minimize=True)
            for config, result in known_trials:
                loss = hyperparameter_loss(config, result)
                _, trial_index = ax_client.attach_trial(dict(config))
                ax_client.complete_trial(
                    trial_index=trial_index,
                    raw_data={"hyperparameter_loss": (loss, None)})

            tune.run(
                insert_hyperparameter_loss(exploring_trainable),
                name=experiment_name,
                search_alg=AxSearch(ax_client,
                                    max_concurrent=(torch.cuda.device_count()
                                                    if torch.cuda.is_available()
                                                    else 1)),
                num_samples=args.num_configs + num_random_remaining,
                resources_per_trial={
                    "cpu": 1,
                    "gpu": (1 if torch.cuda.is_available() else 0)},
                loggers=DEFAULT_LOGGERS,
                verbose=1,
            )

            num_random_remaining = 0

        print("STAGE 2: Make sure we have enough samples for the best config")
        while True:
            best_config, best_results = get_best_config(experiment_dir,
                                                        parameters,
                                                        num_training_iterations)
            if len(best_results) < num_best_config_samples:
                tune.run(
                    followup_trainable,
                    name=experiment_name,
                    config=dict(best_config),
                    num_samples=num_best_config_samples - len(best_results),
                    resources_per_trial={
                        "cpu": 1,
                        "gpu": (1 if torch.cuda.is_available() else 0)},
                    loggers=DEFAULT_LOGGERS,
                    verbose=1,
                )
            else:
                break

    best_config, best_results = get_best_config(experiment_dir,
                                                parameters,
                                                num_training_iterations)
    acc = np.mean([result["mean_accuracy"]
                   for result in best_results])
    output = (best_config, acc)

    output_dir = Path(script_dir) / "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"{experiment_name}.json"
    with open(output_file, "w") as f:
        print(f"Saving {output_file}")
        json.dump(output, f)
