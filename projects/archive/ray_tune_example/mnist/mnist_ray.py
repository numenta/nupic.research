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

import argparse
import configparser
import os

import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.torch.modules import (
    Flatten,
    KWinners,
    KWinners2d,
    SparseWeights,
    rezero_weights,
    update_boost_strength,
)


class TrainMNIST(tune.Trainable):
    """ray.tune trainable class:

    - Override _setup to reset the experiment for each trial.
    - Override _train to train and evaluate each epoch
    - Override _save and _restore to serialize the model
    """

    def _setup(self, config):

        # Get trial parameters
        seed = config["seed"]
        datadir = config["datadir"]
        batch_size = config["batch_size"]
        test_batch_size = config["test_batch_size"]
        first_epoch_batch_size = config["first_epoch_batch_size"]
        in_channels, h, w = config["c1_input_shape"]
        learning_rate = config["learning_rate"]
        momentum = config["momentum"]
        weight_sparsity = config["weight_sparsity"]
        boost_strength = config["boost_strength"]
        boost_strength_factor = config["boost_strength_factor"]
        n = config["n"]
        percent_on = config["percent_on"]
        cnn_percent_on = config["cnn_percent_on"]
        k_inference_factor = config["k_inference_factor"]
        kernel_size = config["kernel_size"]
        out_channels = config["out_channels"]
        output_size = config["output_size"]
        cnn_output_len = out_channels * ((w - kernel_size + 1) // 2) ** 2

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            self.device = torch.device("cpu")

        xforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(datadir, train=True, transform=xforms)
        test_dataset = datasets.MNIST(datadir, train=False, transform=xforms)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True
        )
        self.first_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=first_epoch_batch_size, shuffle=True
        )

        # Create simple sparse model
        self.model = nn.Sequential()

        # CNN layer
        self.model.add_module(
            "cnn",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
        )

        if cnn_percent_on < 1.0:
            self.model.add_module(
                "kwinners_cnn",
                KWinners2d(
                    percent_on=cnn_percent_on,
                    channels=out_channels,
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                ),
            )
        else:
            self.model.add_module("ReLU_cnn", nn.ReLU())

        self.model.add_module("maxpool", nn.MaxPool2d(kernel_size=2))

        # Flatten max pool output before passing to linear layer
        self.model.add_module("flatten", Flatten())

        # Linear layer
        linear = nn.Linear(cnn_output_len, n)
        if weight_sparsity < 1.0:
            self.model.add_module(
                "sparse_linear", SparseWeights(linear, weight_sparsity)
            )
        else:
            self.model.add_module("linear", linear)

        if percent_on < 1.0:
            self.model.add_module(
                "kwinners_kinear",
                KWinners(
                    n=n,
                    percent_on=percent_on,
                    k_inference_factor=k_inference_factor,
                    boost_strength=boost_strength,
                    boost_strength_factor=boost_strength_factor,
                ),
            )
        else:
            self.model.add_module("Linear_ReLU", nn.ReLU())

        # Output layer
        self.model.add_module("fc", nn.Linear(n, output_size))
        self.model.add_module("softmax", nn.LogSoftmax(dim=1))

        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )

    def _train(self):
        if self._iteration == 0:
            train_loader = self.first_loader
        else:
            train_loader = self.train_loader

        train_model(
            model=self.model,
            loader=train_loader,
            optimizer=self.optimizer,
            device=self.device,
            post_batch_callback=self._post_batch,
        )
        self.model.apply(update_boost_strength)

        return evaluate_model(
            model=self.model, loader=self.test_loader, device=self.device
        )

    def _post_batch(self, *args, **kwargs):
        self.model.apply(rezero_weights)

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


@ray.remote
def run_experiment(config, trainable):
    """Run a single tune experiment in parallel as a "remote" function.

    :param config: The experiment configuration
    :type config: dict
    :param trainable: tune.Trainable class with your experiment
    :type trainable: :class:`ray.tune.Trainable`
    """
    # Stop criteria. Default to total number of iterations/epochs
    stop_criteria = {"training_iteration": config.get("iterations")}
    stop_criteria.update(config.get("stop", {}))

    tune.run(
        trainable,
        name=config["name"],
        local_dir=config["path"],
        stop=stop_criteria,
        config=config,
        num_samples=config.get("repetitions", 1),
        search_alg=config.get("search_alg", None),
        scheduler=config.get("scheduler", None),
        trial_executor=config.get("trial_executor", None),
        checkpoint_at_end=config.get("checkpoint_at_end", False),
        checkpoint_freq=config.get("checkpoint_freq", 0),
        resume=config.get("resume", False),
        reuse_actors=config.get("reuse_actors", False),
        verbose=config.get("verbose", 0),
    )


def parse_config(config_file, experiments=None):
    """Parse configuration file optionally filtering for specific
    experiments/sections.

    :param config_file: Configuration file
    :param experiments: Optional list of experiments
    :return: Dictionary with the parsed configuration
    """
    cfgparser = configparser.ConfigParser()
    cfgparser.read_file(config_file)

    params = {}
    for exp in cfgparser.sections():
        if not experiments or exp in experiments:
            values = cfgparser.defaults()
            values.update(dict(cfgparser.items(exp)))
            item = {}
            for k, v in values.items():
                try:
                    item[k] = eval(v)
                except (NameError, SyntaxError):
                    item[k] = v

            params[exp] = item

    return params


def parse_options():
    """parses the command line options for different settings."""
    optparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    optparser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=open,
        default="experiments.cfg",
        help="your experiments config file",
    )
    optparser.add_argument(
        "-n",
        "--num_cpus",
        dest="num_cpus",
        type=int,
        default=os.cpu_count(),
        help="number of cpus you want to use",
    )
    optparser.add_argument(
        "-g",
        "--num_gpus",
        dest="num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="number of gpus you want to use",
    )
    optparser.add_argument(
        "-e",
        "--experiment",
        action="append",
        dest="experiments",
        help="run only selected experiments, by default run all experiments in "
        "config file.",
    )

    return optparser.parse_args()


if __name__ == "__main__":
    # Load and parse command line option and experiment configurations
    options = parse_options()
    configs = parse_config(options.config, options.experiments)

    # Use configuration file location as the project location.
    # Ray Tune default working directory is "~/ray_results"
    project_dir = os.path.dirname(options.config.name)
    project_dir = os.path.abspath(project_dir)

    # Pre-download dataset
    datadir = os.path.join(project_dir, "data")
    train_dataset = datasets.MNIST(datadir, download=True, train=True)

    # Initialize ray cluster
    ray.init(
        num_cpus=options.num_cpus,
        num_gpus=options.num_gpus,
        local_mode=options.num_cpus == 1,
    )

    # Run all experiments in parallel
    results = []
    for exp in configs:
        config = configs[exp]
        config["name"] = exp

        # Make sure local directories are relative to the project location
        path = config.get("path", "results")
        if not os.path.isabs(path):
            config["path"] = os.path.join(project_dir, path)

        datadir = config.get("datadir", "data")
        if not os.path.isabs(datadir):
            config["datadir"] = os.path.join(project_dir, datadir)

        # When running multiple hyperparameter searches on different experiments,
        # ray.tune will run one experiment at the time. We use "ray.remote" to
        # run each tune experiment in parallel as a "remote" function and wait until
        # all experiments complete
        results.append(run_experiment.remote(config, TrainMNIST))

    # Wait for all experiments to complete
    ray.get(results)

    ray.shutdown()
