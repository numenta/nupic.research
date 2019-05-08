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
import json
import os

import click
import matplotlib
import pandas as pd
from torchvision import datasets

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nupic.research.support.parse_config import parse_config
from projects.whydense.cifar.mobilenet_cifar import MobileNetCIFAR10

# Used by "parse_config"
import torch
import torch.nn as nn
from nupic.research.frameworks.pytorch.models import *


def trainModels(configs, projectDir):
  for exp in configs:
    config = configs[exp]
    config["name"] = exp

    # Make sure local directories are relative to the project location
    path = config.get("path", "results")
    if not os.path.isabs(path):
      path = os.path.join(projectDir, path)

    # Separate path for each experiment
    path = os.path.join(path, exp)
    config["path"] = path

    data_dir = config.get("data_dir", "data")
    if not os.path.isabs(data_dir):
      config["data_dir"] = os.path.join(projectDir, data_dir)

    experiment = MobileNetCIFAR10(config)
    os.makedirs(experiment.config["path"], exist_ok=True)

    # Train model and collect results
    checkpoint_freq = experiment.config["checkpoint_freq"]
    results = []
    for epoch in range(1, experiment.config["iterations"] + 1):
      experiment.train(epoch)
      results.append(experiment.test(epoch))
      if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
        experiment.save(path)

    # Save model
    if experiment.config["checkpoint_at_end"]:
      experiment.save(path)

    with open(os.path.join(experiment.config["path"], "results.json"), "w") as f:
      json.dump(results, f)

    # Save results
    df = pd.DataFrame(results)
    filename = os.path.join(experiment.config["path"], "results")
    df.to_csv(filename + ".csv", index=False)
    df.to_json(filename + ".json")

    # Plot
    plt.figure()
    df["mean_accuracy"].plot()
    plt.savefig(filename + "_accuracy.pdf")
    plt.close()

    plt.figure()
    df["mean_loss"].plot()
    plt.savefig(filename + "_loss.pdf")
    plt.close()

    # Save domino board stats
    best_epoch = df['mean_accuracy'].idxmax()
    row = df.loc[best_epoch]
    row["epoch"] = best_epoch
    row.to_json("dominostats.json")



@click.command()
@click.option("-c", "--config", type=open, default="mobilenet_experiments.cfg",
              show_default=True, help="your experiments config file")
@click.option("-e", "--experiment", multiple=True,
              help="run only selected experiments, by default run all "
                   "experiments in config file.")
@click.option("-l", "--list", "show_list", is_flag=True,
              help="show list of available experiments.")
def main(config, experiment, show_list):
  configs = parse_config(config, experiment, globals=globals())
  if show_list:
    print("Experiments:", list(configs.keys()))
    return

  projectDir = os.path.dirname(config.name)
  projectDir = os.path.abspath(projectDir)

  # Load dataset
  data_dir = os.path.join(projectDir, "data")
  datasets.CIFAR10(data_dir, download=True, train=True)

  trainModels(configs, projectDir=projectDir)



if __name__ == "__main__":
  main()
