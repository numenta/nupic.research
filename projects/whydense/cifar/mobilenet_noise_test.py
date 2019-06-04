#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import subprocess
from collections import OrderedDict, defaultdict

import click
import matplotlib
import matplotlib.pyplot as plt
import ray
import ray.tune as tune
import torch
from torchvision import datasets

from nupic.research.support import load_ray_tune_experiment, parse_config
from projects.whydense.cifar.mobilenet_cifar import MobileNetCIFAR10

matplotlib.use("Agg")


NOISE_VALUES = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175)


def download_s3_results(s3url, local_dir):
    """Download results from S3 using AWS CLI.

    :param s3url: The S3 URI where to download the results from
    :param local_dir: Where to put the results
    """
    subprocess.run(["aws", "s3", "sync", s3url, local_dir], check=True)


class MobileNetNoiseTune(tune.Trainable):
    """ray.tune trainable class for running MobileNet CIFAR experiments:"""

    def _setup(self, config):
        # Local imports passed to the "eval" function
        import torch.nn as nn  # noqa F401
        from nupic.research.frameworks.pytorch.models import (  # noqa F401
            MobileNetV1,
            mobile_net_v1_sparse_point,
            mobile_net_v1_sparse_depth,
        )

        # Cannot pass functions to "ray.tune". Make sure to use string in the config
        # and evaluate during "_setup"
        model_config = config["model_config"]
        model_config["loss_function"] = eval(
            model_config["loss_function"], globals(), locals()
        )
        model_config["model_type"] = eval(
            model_config["model_type"], globals(), locals()
        )
        model_config["noise"] = config["noise"]
        checkpoint_path = config["checkpoint_path"]
        self.experiment = MobileNetCIFAR10(model_config)
        self.experiment.restore(checkpoint_path)

    def _train(self):
        return self.experiment.test()


@ray.remote
def run_experiment(config, trainable, num_cpus=1, num_gpus=0):
    """Run a single tune experiment in parallel as a "remote" function.

    :param config: The experiment configuration
    :type config: dict
    :param trainable_cls: tune.Trainable class with your experiment
    :type trainable_cls: :class:`ray.tune.Trainable`
    """
    resources_per_trial = {"cpu": num_cpus, "gpu": num_gpus}
    print("experiment =", config["name"])
    print("resources_per_trial =", resources_per_trial)
    print("config =", config)

    # Stop criteria. Default to total number of iterations/epochs
    stop_criteria = {"training_iteration": config.get("iterations")}
    stop_criteria.update(config.get("stop", {}))
    print("stop_criteria =", stop_criteria)

    return tune.run(
        trainable,
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


@click.group(chain=True)
def cli():
    pass


@cli.command(help="Plot noise experiments")
@click.option(
    "-c",
    "--config",
    type=open,
    default="mobilenet_experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiments",
    multiple=True,
    help="run only selected experiments, by default run all "
    "experiments in config file.",
)
def plot(config, experiments):
    print("config =", config.name)
    print("experiments =", experiments)

    # Use configuration file location as the project location.
    project_dir = os.path.dirname(config.name)
    project_dir = os.path.abspath(project_dir)
    print("project_dir =", project_dir)

    # Load and parse experiment configurations
    noise_experiments = defaultdict(list)
    configs = parse_config(config, experiments, globals(), locals())
    for exp in configs:
        config = configs[exp]
        # Load experiment state and get all the tags
        experiment_path = os.path.join(project_dir, config["path"], exp)
        experiment_state = load_ray_tune_experiment(experiment_path)
        all_experiments = experiment_state["checkpoints"]
        for experiment in all_experiments:
            noise_experiments[exp].append(experiment["experiment_tag"])

    # Plot noise experiments
    for exp in noise_experiments:
        fig, ax = plt.subplots()

        for tag in noise_experiments[exp]:
            # Load experiment results
            experiment_path = os.path.join(project_dir, "results", "noise", exp, tag)
            if not os.path.exists(experiment_path):
                continue

            experiment_state = load_ray_tune_experiment(
                experiment_path, load_results=True
            )

            all_experiments = experiment_state["checkpoints"]
            data = {}
            for experiment in all_experiments:
                acc = experiment["results"][0]["mean_accuracy"]
                noise = experiment["config"]["noise"]
                data.setdefault(noise, acc)
            data = OrderedDict(sorted(data.items(), key=lambda i: i[0]))
            ax.plot(list(data.keys()), list(data.values()), label=tag)

        fig.suptitle("Accuracy vs noise")
        ax.set_xlabel("Noise")
        ax.set_ylabel("Accuracy (percent)")
        plt.legend()
        plt.grid(axis="y")
        plot_path = os.path.join(
            project_dir, "results", "noise", "{}_noise.pdf".format(exp)
        )
        plt.savefig(plot_path)
        plt.close()


@cli.command(help="Run noise experiments on the best scoring model")
@click.option(
    "-c",
    "--config",
    type=open,
    default="mobilenet_experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiments",
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
@click.option(
    "--redis-address",
    help="Optional ray cluster redis address",
    show_envvar=True,
    envvar="REDIS_ADDRESS",
)
@click.option(
    "-v",
    "--noise-values",
    multiple=True,
    default=NOISE_VALUES,
    show_default=True,
    help="Noise values",
    type=float,
)
def run(config, experiments, num_cpus, num_gpus, redis_address, noise_values):
    print("config =", config.name)
    print("experiments =", experiments)
    print("num_gpus =", num_gpus)
    print("num_cpus =", num_cpus)
    print("redis_address =", redis_address)
    print("noise_values =", noise_values)

    # Use configuration file location as the project location.
    project_dir = os.path.dirname(config.name)
    project_dir = os.path.abspath(project_dir)
    print("project_dir =", project_dir)

    # Download dataset
    data_dir = os.path.join(project_dir, "data")
    datasets.CIFAR10(data_dir, download=True, train=True)

    # Initialize ray cluster
    if redis_address is not None:
        ray.init(redis_address=redis_address, include_webui=True)
    else:
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=num_cpus == 1)

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals(), locals())

    # Run all experiments in parallel
    ray_trials = []
    for exp in configs:
        config = configs[exp]

        # noise experiment tune configuration
        noise_config = {"iterations": 1, "noise": {"grid_search": list(noise_values)}}

        # Download results from S3 when running on the cluster
        if redis_address is not None and "upload_dir" in config:
            upload_dir = config["upload_dir"]
            download_s3_results(
                "{}/{}".format(upload_dir, exp), os.path.join(config["path"], exp)
            )

            # Store noise results with original results in S3
            noise_config["upload_dir"] = "{}".format(upload_dir)
            noise_config[
                "sync_function"
            ] = "aws s3 sync `dirname {local_dir}` {remote_dir}/`basename $(dirname {local_dir})`"  # noqa E501
        else:
            noise_config.pop("upload_dir", None)
            noise_config.pop("sync_function", None)

        # Load experiment results
        experiment_path = os.path.join(project_dir, config["path"], exp)
        experiment_state = load_ray_tune_experiment(experiment_path, load_results=True)
        all_experiments = experiment_state["checkpoints"]
        for experiment in all_experiments:
            # Make logs relative to experiment path
            logdir = experiment["logdir"]
            logpath = os.path.join(experiment_path, os.path.basename(logdir))

            # Check for experiment results
            results = experiment["results"]
            if results is None:
                continue

            # Get best scoring model checkpoint from results
            best_result = max(results, key=lambda x: x["mean_accuracy"])

            epoch = best_result["training_iteration"]
            checkpoint_path = os.path.join(logpath, "checkpoint_{}".format(epoch))
            if os.path.exists(checkpoint_path):
                # Update data path
                model_config = best_result["config"]
                model_config["data_dir"] = data_dir

                # Run noise tests
                noise_config.update(
                    {
                        "name": experiment["experiment_tag"],
                        "path": os.path.join(project_dir, "results", "noise", exp),
                        "checkpoint_path": checkpoint_path,
                        "model_config": model_config,
                    }
                )

                ray_trials.append(
                    run_experiment.remote(
                        noise_config,
                        MobileNetNoiseTune,
                        num_cpus=1,
                        num_gpus=min(1, num_gpus),
                    )
                )

    # Wait for all experiments to complete
    ray.get(ray_trials)
    ray.shutdown()


if __name__ == "__main__":
    cli()
