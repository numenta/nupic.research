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
import subprocess
from pathlib import Path

import click
import numpy as np  # noqa F401
import ray
import ray.tune as tune
import torch

from nupic.research.frameworks.pytorch.model_utils import set_random_seed
from nupic.research.support import load_ray_tune_experiment, parse_config
from sparse_speech_experiment import SparseSpeechExperiment


class SpeechExperimentTune(tune.Trainable):
    """Ray tune trainable wrapping MNISTSparseExperiment."""

    def _setup(self, config):
        self.experiment = SparseSpeechExperiment(config)

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
        best_config = best_result["config"]

        # Update path
        best_config["path"] = config["path"]
        best_config["data_dir"] = config["data_dir"]

        # Load pre-trained model from checkpoint and run noise test on it
        logdir = os.path.join(experiment_path, os.path.basename(checkpoint["logdir"]))
        checkpoint_path = os.path.join(logdir, "checkpoint_{}".format(best_epoch))
        experiment = SparseSpeechExperiment(best_config)
        experiment.restore(checkpoint_path)

        # Save noise results in checkpoint log dir
        noise_test = os.path.join(logdir, "noise.json")
        with open(noise_test, "w") as f:
            res = experiment.run_noise_tests()
            json.dump(res, f)
            print(res)

        # Compute total noise score
        total_correct = 0
        for k, v in res.items():
            print(k, v, v["total_correct"])
            total_correct += v["total_correct"]
        print("Total across all noise values", total_correct)

    # Upload results to S3
    sync_function = config.get("sync_function", None)
    if sync_function is not None:
        upload_dir = config["upload_dir"]
        final_cmd = sync_function.format(
            local_dir=experiment_path, remote_dir=upload_dir
        )
        subprocess.Popen(final_cmd, shell=True)


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
    type=float,
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

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals_param=globals())

    if show_list:
        print("Experiments:", list(configs.keys()))
        return

    # Initialize ray cluster
    if redis_address is not None:
        ray.init(redis_address=redis_address, include_webui=True)
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=num_cpus == 1)

    # Run experiments
    gpu_percent = 0
    if num_gpus > 0:
        gpu_percent = configs.get("gpu_percentage", 0.5)
    resources_per_trial = {"cpu": 1, "gpu": gpu_percent}
    print("resources_per_trial =", resources_per_trial)
    for exp in configs:
        print("experiment =", exp)
        config = configs[exp]
        config["name"] = exp

        # Stop criteria. Default to total number of iterations/epochs
        stop_criteria = {"training_iteration": config.get("iterations")}
        stop_criteria.update(config.get("stop", {}))
        print("stop_criteria =", stop_criteria)

        # Make sure path and data_dir are relative to the project location,
        # handling both ~/nta and ../results style paths.
        path = config.get("path", ".")
        config["path"] = str(Path(path).expanduser().resolve())

        data_dir = config.get("data_dir", "data")
        config["data_dir"] = str(Path(data_dir).expanduser().resolve())

        tune.run(
            SpeechExperimentTune,
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

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals_param=globals())

    # Initialize ray cluster
    if redis_address is not None:
        ray.init(redis_address=redis_address, include_webui=True)
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=num_cpus == 1)

    # FIXME: Update remote function resource usage
    num_gpus = float(num_gpus / num_cpus)
    run_noise_test._num_gpus = num_gpus
    run_noise_test.num_cpus = 1

    # Run experiments
    results = []
    for exp in configs:
        config = configs[exp]
        config["name"] = exp

        # Make sure path and data_dir are relative to the project location,
        # handling both ~/nta and ../results style paths.
        path = config.get("path", ".")
        config["path"] = str(Path(path).expanduser().resolve())

        data_dir = config.get("data_dir", "data")
        config["data_dir"] = str(Path(data_dir).expanduser().resolve())

        # Run each experiment in parallel
        results.append(run_noise_test.remote(config))

    # Wait until all experiments complete
    ray.get(results)
    ray.shutdown()

    print(results)


if __name__ == "__main__":
    set_random_seed(18)
    cli()
