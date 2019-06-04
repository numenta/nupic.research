# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import json
import os

import click
import numpy as np  # noqa F401
import ray
import ray.tune as tune
import torch
from torchvision import datasets

from mnist_sparse_experiment import MNISTSparseExperiment
from nupic.research.frameworks.pytorch.model_utils import set_random_seed
from nupic.research.support import load_ray_tune_experiment, parse_config


class MNISTExperimentTune(tune.Trainable):
    """Ray tune trainable wrapping MNISTSparseExperiment."""

    def _setup(self, config):
        self.experiment = MNISTSparseExperiment(config)

    def _train(self):
        self.experiment.train(self._iteration)
        return self.experiment.test()

    def _save(self, checkpoint_dir):
        return self.experiment.save(checkpoint_dir)

    def _restore(self, checkpoint_dir):
        self.experiment.restore(checkpoint_dir)


@ray.remote
def run_noise_test(config):
    """Run noise test on the best scoring model found during training. Make
    sure to train the models before calling this function.

    :param config: The configuration of the pre-trained model.
    :return: dict with noise test results over all experiments
    """
    # Load experiment data
    name = config["name"]
    experiment_path = os.path.join(config["path"], name)
    experiment_state = load_ray_tune_experiment(
        experiment_path=experiment_path, load_results=True
    )

    # Go through all checkpoints in the experiment
    all_checkpoints = experiment_state["checkpoints"]
    for checkpoint in all_checkpoints:
        results = checkpoint["results"]
        if results is None:
            continue

        # For each checkpoint select the epoch with the best accuracy as the best epoch
        best_result = max(results, key=lambda x: x["mean_accuracy"])
        best_epoch = best_result["training_iteration"]

        # Load pre-trained model from checkpoint and run noise test on it
        logdir = os.path.join(experiment_path, os.path.basename(checkpoint["logdir"]))
        checkpoint_path = os.path.join(logdir, "checkpoint_{}".format(best_epoch))
        experiment = MNISTSparseExperiment(config)
        experiment.restore(checkpoint_path)

        # Save noise results in checkpoint log dir
        noise_test = os.path.join(logdir, "noise.json")
        with open(noise_test, "w") as f:
            json.dump(experiment.run_noise_tests(), f)


@click.group(chain=True)
def cli():
    pass


@cli.command(help="Train models")
@click.option(
    "-c",
    "--config",
    type=open,
    default="experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiment",
    "experiments",
    multiple=True,
    help="run only selected experiments, by default run all "
    "experiments in config file.",
)
@click.option(
    "-l",
    "--list",
    "show_list",
    is_flag=True,
    help="show list of available experiments.",
)
@click.option(
    "-n",
    "--num_cpus",
    type=int,
    default=os.cpu_count(),
    show_default=True,
    help="number of cpus you want to use",
)
@click.option(
    "-g",
    "--num_gpus",
    type=int,
    default=torch.cuda.device_count(),
    show_default=True,
    help="number of gpus you want to use",
)
@click.option("--redis-address", help="Ray Cluster redis address")
def train(config, experiments, num_cpus, num_gpus, redis_address, show_list):
    print("config =", config.name)
    print("num_gpus =", num_gpus)
    print("num_cpus =", num_cpus)
    print("redis_address =", redis_address)

    # Use configuration file location as the project location.
    project_dir = os.path.dirname(config.name)
    project_dir = os.path.abspath(project_dir)
    print("project_dir =", project_dir)

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals_param=globals())

    if show_list:
        print("Experiments:", list(configs.keys()))
        return

    print("experiments =", list(configs.keys()))

    # download dataset
    data_dir = os.path.join(project_dir, "data")
    datasets.MNIST(data_dir, download=True, train=True)

    # Initialize ray cluster
    if redis_address is not None:
        ray.init(redis_address=redis_address, include_webui=True)
        num_cpus = 1
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=num_cpus == 1)

    # Run experiments
    resources_per_trial = {"cpu": 1, "gpu": num_gpus / num_cpus}
    print("resources_per_trial =", resources_per_trial)
    for exp in configs:
        print("experiment =", exp)
        config = configs[exp]
        config["name"] = exp

        # Stop criteria. Default to total number of iterations/epochs
        stop_criteria = {"training_iteration": config.get("iterations")}
        stop_criteria.update(config.get("stop", {}))
        print("stop_criteria =", stop_criteria)

        # Make sure local directories are relative to the project location
        path = config.get("path", None)
        if path and not os.path.isabs(path):
            config["path"] = os.path.join(project_dir, path)

        data_dir = config.get("data_dir", "data")
        if not os.path.isabs(data_dir):
            config["data_dir"] = os.path.join(project_dir, data_dir)

        tune.run(
            MNISTExperimentTune,
            name=config["name"],
            stop=stop_criteria,
            config=config,
            resources_per_trial=resources_per_trial,
            num_samples=config.get("repetitions", 1),
            local_dir=config.get("path", None),
            upload_dir=config.get("upload_dir", None),
            sync_function=config.get("sync_function", None),
            checkpoint_freq=config.get("checkpoint_freq", 0),
            checkpoint_at_end=config.get("checkpoint_at_end", False),
            export_formats=config.get("", None),
            search_alg=config.get("search_alg", None),
            scheduler=config.get("scheduler", None),
            verbose=config.get("verbose", 2),
            resume=config.get("resume", False),
            queue_trials=config.get("queue_trials", False),
            reuse_actors=config.get("reuse_actors", False),
            trial_executor=config.get("trial_executor", None),
            raise_on_failed_trial=config.get("raise_on_failed_trial", True),
        )

    ray.shutdown()


@cli.command(help="Run noise tests")
@click.option(
    "-c",
    "--config",
    type=open,
    default="experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiment",
    "experiments",
    multiple=True,
    help="run only selected experiments, by default run all "
    "experiments in config file.",
)
@click.option(
    "-n",
    "--num_cpus",
    type=int,
    default=os.cpu_count(),
    show_default=True,
    help="number of cpus you want to use",
)
@click.option(
    "-g",
    "--num_gpus",
    type=int,
    default=torch.cuda.device_count(),
    show_default=True,
    help="number of gpus you want to use",
)
@click.option("--redis-address", help="Ray Cluster redis address")
def noise(config, experiments, num_cpus, num_gpus, redis_address):
    print("config =", config.name)
    print("num_gpus =", num_gpus)
    print("num_cpus =", num_cpus)
    print("redis_address =", redis_address)

    # Use configuration file location as the project location.
    project_dir = os.path.dirname(config.name)
    project_dir = os.path.abspath(project_dir)
    print("project_dir =", project_dir)

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals_param=globals())
    print("experiments =", list(configs.keys()))

    # download dataset
    data_dir = os.path.join(project_dir, "data")
    datasets.MNIST(data_dir, download=True, train=True)

    # Initialize ray cluster
    if redis_address is not None:
        ray.init(redis_address=redis_address, include_webui=True)
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=num_cpus == 1)

    # Run experiments
    results = []
    for exp in configs:
        config = configs[exp]
        config["name"] = exp

        # Make sure local directories are relative to the project location
        path = config.get("path", None)
        if path and not os.path.isabs(path):
            config["path"] = os.path.join(project_dir, path)

        data_dir = config.get("data_dir", "data")
        if not os.path.isabs(data_dir):
            config["data_dir"] = os.path.join(project_dir, data_dir)

        # Avoid "tune.sample_from"
        config["seed"] = 18

        # Run each experiment in parallel
        results.append(run_noise_test.remote(config))

    # Wait until all experiments complete
    ray.get(results)
    ray.shutdown()


if __name__ == "__main__":
    set_random_seed(18)
    cli()
