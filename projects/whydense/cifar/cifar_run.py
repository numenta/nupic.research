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
import json
import os

import torch
from torchvision import datasets

from nupic.research.support.parse_config import parse_config
from projects.whydense.cifar.cifar_experiment import TinyCIFAR


def train_models(configs, project_dir):
    """Run all the training experiments specified in configs."""
    # download dataset
    data_dir = os.path.join(project_dir, "data")
    datasets.CIFAR10(data_dir, download=True, train=True)

    # Run all experiments in serial
    if len(configs) == 0:
        print("No experiments to run!")

    for exp in configs:
        config = configs[exp]
        if "name" not in config:
            config["name"] = exp

        # Make sure local directories are relative to the project location
        path = config.get("path", "results")
        if not os.path.isabs(path):
            config["path"] = os.path.join(project_dir, path)

        data_dir = config.get("data_dir", "data")
        if not os.path.isabs(data_dir):
            config["data_dir"] = os.path.join(project_dir, data_dir)

        model = TinyCIFAR()
        model.model_setup(config)
        for epoch in range(config["iterations"]):
            ret = model.train_epoch(epoch)
            print("epoch=", epoch, ":", ret)
            # if ret['stop'] == 1:
            #   print("Stopping early!")
            #   break

        if config.get("checkpoint_at_end", False):
            model.model_save(path)


def parse_options():
    """parses the command line options for different settings."""
    optparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    optparser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default="",
        help="your experiment config file",
    )
    optparser.add_argument(
        "-p",
        "--params",
        dest="params",
        type=str,
        default="",
        help="your experiment params json file",
    )
    optparser.add_argument(
        "-e",
        "--experiment",
        action="append",
        dest="experiments",
        help="run only selected experiments, by default run "
        "all experiments in config file.",
    )

    return optparser.parse_args()


if __name__ == "__main__":

    print("Using torch version", torch.__version__)
    print("Torch device count=", torch.cuda.device_count())
    # Load and parse command line option and experiment configurations
    options = parse_options()
    if options.config != "":
        with open(options.config) as f:
            configs = parse_config(f, options.experiments)
        project_dir = os.path.dirname(options.config)

    elif options.params != "":
        with open(options.params) as f:
            params = json.load(f)
            params["data_dir"] = os.path.abspath(os.path.join(".", "data"))
            params["path"] = os.path.abspath(os.path.dirname(options.params))
            configs = {params["name"]: params}
        project_dir = "."

    else:
        raise RuntimeError("Either a .cfg or a params .json file must be specified")

    # Use configuration file location as the project location.
    project_dir = os.path.abspath(project_dir)

    train_models(configs, project_dir=project_dir)
