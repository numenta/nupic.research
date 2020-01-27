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
    filter_to_pareto,
    get_ray_trials,
)
from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)


def hyperparameter_loss(config, result, alpha):
    log_error = np.log(np.maximum(1 - result["mean_accuracy"], 1e-7))
    log_nz = np.log(np.maximum(result["inference_nz"], 1))
    return (alpha * log_error) + ((1 - alpha) * log_nz)


def get_frontier_trials(experiment_dir,
                        parameters,
                        num_training_iterations):
    known_trials = get_ray_trials(experiment_dir, parameters)
    known_trials = [
        (config, result)
        for config, result in known_trials
        if result["training_iteration"] == num_training_iterations]

    grouped_trials = defaultdict(list)
    for config, result in known_trials:
        grouped_trials[config].append(result)
    grouped_trials = list(grouped_trials.items())

    frontier_trials = filter_to_pareto(grouped_trials, [
        lambda config, results: np.mean([1 - result["mean_accuracy"]
                                         for result in results]),
        lambda config, results: np.mean([result["inference_nz"]
                                         for result in results])
    ])

    # Don't waste time on results that are ~no better than random.
    frontier_trials = [(config, results)
                       for config, results in frontier_trials
                       if np.mean([result["mean_accuracy"]
                                   for result in results]) > 0.2]

    return frontier_trials


def insert_hyperparameter_loss(trainable):
    class Cls(trainable):
        def _setup(self, config):
            self.alpha = config["alpha"]
            config2 = dict(config)
            del config2["alpha"]
            self.config = config2
            super()._setup(config2)

        def _train(self):
            result = super()._train()
            result["hyperparameter_loss"] = hyperparameter_loss(
                self.config, result, self.alpha)
            return result

    Cls.__name__ = trainable.__name__

    return Cls


def ax_optimize_accuracy_weightsparsity(
        exploring_trainable, followup_trainable, experiment_name, script_dir,
        alphas, parameters, num_training_iterations,
        samples_per_frontier_config):
    """
    Optimize a Ray Trainable's "mean_accuracy" and "inference_nz", using Ax to
    select hyperparameters. This procedure will pick up any existing results for
    this experiment and plug them into the Ax model, then will generate new
    results. To optimize two objectives, this procedure chooses a set of
    projections from two scalar objectives to a single scalar objective. These
    projections are defined by alpha*log(error) + (1 - alpha)*log(inference_nz).
    The procedure loops over the different alphas and rebuilds the Ax model and
    generates new results for each. All results go into the same Ray experiment
    folder, so each search (i.e. each alpha) benefits from all of the other
    searches.
    """

    experiment_dir = os.path.expanduser(f"~/ray_results/{experiment_name}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-initial-configs", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--num-random", type=int, default=16)
    parser.add_argument("--num-configs", type=int, default=16)
    args = parser.parse_args()

    ray.init()

    if args.run_initial_configs:
        input_dir = Path(script_dir) / "best_configs"
        input_file = input_dir / f"{experiment_name}.json"
        if input_file.exists():
            print("STAGE 0: Test initial suggested hyperparameters")
            with open(input_file, "r") as f:
                frontier_trials = json.load(f)

            configs = [dict(config)
                       for config, _, _ in frontier_trials]
            # Only include parameters specified in the parameters list.
            configs = [{parameter["name"]: config[parameter["name"]]
                        for parameter in parameters}
                       for config in configs]

            tune.run(
                exploring_trainable,
                name=experiment_name,
                config=tune.grid_search(configs),
                num_samples=1,
                resources_per_trial={
                    "cpu": 1,
                    "gpu": (1 if torch.cuda.is_available() else 0)},
                loggers=DEFAULT_LOGGERS,
                verbose=1,
            )

    num_random_remaining = args.num_random

    for _ in range(args.num_iterations):
        print("STAGE 1: Hyperparameter search")
        for alpha in alphas:
            if num_random_remaining + args.num_configs == 0:
                break

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
                loss = hyperparameter_loss(config, result, alpha)
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
                config=dict(alpha=alpha),
                num_samples=args.num_configs + num_random_remaining,
                resources_per_trial={
                    "cpu": 1,
                    "gpu": (1 if torch.cuda.is_available() else 0)},
                loggers=DEFAULT_LOGGERS,
                verbose=1,
            )

            num_random_remaining = 0

        # Verify the frontier
        print("STAGE 2: Make sure we have enough samples for each point at the "
              "frontier")
        while True:
            resample_configs = []
            for config, results in get_frontier_trials(experiment_dir,
                                                       parameters,
                                                       num_training_iterations):
                if len(results) < samples_per_frontier_config:
                    resample_configs += (samples_per_frontier_config
                                         - len(results)) * [dict(config)]

            if len(resample_configs) > 0:
                tune.run(
                    followup_trainable,
                    name=experiment_name,
                    config=tune.grid_search(resample_configs),
                    num_samples=1,
                    resources_per_trial={
                        "cpu": 1,
                        "gpu": (1 if torch.cuda.is_available() else 0)},
                    loggers=DEFAULT_LOGGERS,
                    verbose=1,
                )
            else:
                break

    output = []
    for config, results in get_frontier_trials(experiment_dir,
                                               parameters,
                                               num_training_iterations):
        acc = np.mean([result["mean_accuracy"]
                       for result in results])
        nz = np.mean([result["inference_nz"]
                      for result in results])
        output.append((config, acc, nz))

    output_dir = Path(script_dir) / "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / f"{experiment_name}.json"
    with open(output_file, "w") as f:
        print(f"Saving {output_file}")
        json.dump(output, f)
