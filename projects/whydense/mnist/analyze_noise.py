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
import logging
from os.path import basename, dirname
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.image_transforms import RandomNoise
from nupic.research.support import (
    load_ray_tune_experiment, parse_config,
)


logging.basicConfig(level=logging.ERROR)


matplotlib.use("Agg")


NOISE_VALUES = [
    "0.0",
    "0.05",
    "0.1",
    "0.15",
    "0.2",
    "0.25",
    "0.3",
    "0.35",
    "0.4",
    "0.45",
    "0.5",
]

EXPERIMENTS = {
    "denseCNN1": {"label": "dense-CNN1", "linestyle": "--", "marker": "o"},
    "denseCNN2": {"label": "dense-CNN2", "linestyle": "--", "marker": "x"},
    "sparseCNN1": {"label": "sparse-CNN1", "linestyle": "-", "marker": "*"},
    "sparseCNN2": {"label": "sparse-CNN2", "linestyle": "-", "marker": "x"},
}


def plot_noise_curve(configs, results, plot_path):
    fig, ax = plt.subplots()
    fig.suptitle("Accuracy vs noise")
    ax.set_xlabel("Noise")
    ax.set_ylabel("Accuracy (percent)")
    for exp in configs:
        ax.plot(results[exp], **EXPERIMENTS[exp])

    # ax.xaxis.set_ticks(np.arange(0.0, 0.5 + 0.1, 0.1))
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(plot_path)
    plt.close()


def plot_images_with_noise(datadir, noise_values, plot_path):
    """Plot Sample MNIST images with noise."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(datadir, train=False, download=True, transform=transform)

    num_noise = len(noise_values)
    fig = plt.figure(figsize=(num_noise, 4))
    for y in range(4):
        for x in range(num_noise):
            transform.transforms.append(
                RandomNoise(noise_values[x], high_value=0.1307 + 2 * 0.3081)
            )
            img, _ = dataset[y]
            transform.transforms.pop()

            ax = fig.add_subplot(4, num_noise, y * num_noise + x + 1)
            ax.set_axis_off()
            ax.imshow(img.numpy().reshape((28, 28)), cmap="gray")
            if y == 0:
                ax.set_title("{0}%".format(noise_values[x] * 100))

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


@click.command()
@click.option(
    "-c",
    "--config",
    metavar="FILE",
    type=open,
    default="experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
def main(config):
    # Use configuration file location as the project location.
    project_dir = Path(dirname(config.name)).expanduser().resolve()
    data_dir = Path(project_dir) / "data"

    # Load and parse experiment configurations
    configs = parse_config(
        config_file=config,
        experiments=list(EXPERIMENTS.keys()),
        globals_param=globals(),
    )

    results = {}
    for exp in configs:
        config = configs[exp]

        # Load experiment data
        data_dir = Path(config["data_dir"]).expanduser().resolve()
        path = Path(config["path"]).expanduser().resolve()

        experiment_path = path / exp
        experiment_state = load_ray_tune_experiment(
            experiment_path=experiment_path, load_results=True
        )

        # Load noise score and compute the mean_accuracy over all checkpoints
        exp_df = pd.DataFrame()
        for checkpoint in experiment_state["checkpoints"]:
            logdir = experiment_path / basename(checkpoint["logdir"])
            filename = logdir / "noise.json"
            with open(filename, "r") as f:
                df = pd.DataFrame(json.load(f)).transpose()
                exp_df = exp_df.append(df["mean_accuracy"], ignore_index=True)

        results[exp] = exp_df.mean()

    plot_path = project_dir / "accuracy_vs_noise.pdf"
    plot_noise_curve(configs=configs, results=results, plot_path=plot_path)

    # Plot noisy images
    plot_path = project_dir / "mnist_images_with_noise.pdf"
    plot_images_with_noise(
        datadir=data_dir,
        noise_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()
