# -*- coding: utf-8 -*-
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
import re
from itertools import groupby
from pathlib import Path

import click
import numpy as np
from tabulate import tabulate

from nupic.research.support import load_ray_tune_experiment, parse_config


@click.command(help="Train models")
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
    "-f",
    "--format",
    "tablefmt",
    help="Table format",
    type=click.Choice(choices=["grid", "latex"]),
    show_default=True,
    default="grid",
)
def main(config, experiments, tablefmt):
    # Use configuration file location as the project location.
    project_dir = os.path.dirname(config.name)
    project_dir = os.path.abspath(project_dir)
    print("project_dir =", project_dir)
    test_scores_table = [["Network", "Test Score", "Noise Score"]]

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals_param=globals())

    # Select tags ignoring seed value
    def key_func(x):
        s = re.split("[,_]", re.sub(",|\\d+_|seed=\\d+", "", x["experiment_tag"]))
        if len(s[0]) == 0:
            return [" "]
        return s

    for exp in configs:
        config = configs[exp]

        # Make sure path and data_dir are relative to the project location,
        # handling both ~/nta and ../results style paths.
        path = config.get("path", ".")
        config["path"] = str(Path(path).expanduser().resolve())

        data_dir = config.get("data_dir", "data")
        config["data_dir"] = str(Path(data_dir).expanduser().resolve())

        # Load experiment data
        experiment_path = os.path.join(config["path"], exp)
        experiment_state = load_ray_tune_experiment(
            experiment_path=experiment_path, load_results=True
        )

        # Go through all checkpoints in the experiment
        all_checkpoints = experiment_state["checkpoints"]

        # Group checkpoints by tags
        checkpoint_groups = {
            k[0]: list(v)
            for k, v in groupby(sorted(all_checkpoints, key=key_func), key=key_func)
        }

        for tag in checkpoint_groups:
            checkpoints = checkpoint_groups[tag]
            num_exps = len(checkpoints)
            test_scores = np.zeros(num_exps)
            noise_scores = np.zeros(num_exps)

            for i, checkpoint in enumerate(checkpoints):
                results = checkpoint["results"]
                if results is None:
                    continue

                # For each checkpoint select the epoch with the best accuracy as
                # the best epoch
                best_result = max(results, key=lambda x: x["mean_accuracy"])
                test_scores[i] = best_result["mean_accuracy"]

                # Load noise score
                logdir = os.path.join(
                    experiment_path, os.path.basename(checkpoint["logdir"])
                )
                filename = os.path.join(logdir, "noise.json")
                if os.path.exists(filename):
                    with open(filename, "r") as f:
                        noise = json.load(f)

                    noise_scores[i] = sum(x["total_correct"] for x in list(noise.values()))

            test_score = "{0:.2f} ± {1:.2f}".format(
                test_scores.mean(), test_scores.std()
            )
            noise_score = "{0:,.0f} ± {1:.2f}".format(
                noise_scores.mean(), noise_scores.std()
            )
            test_scores_table.append(
                ["{} {}".format(exp, tag), test_score, noise_score]
            )

    print()
    print(tabulate(test_scores_table, headers="firstrow", tablefmt=tablefmt))


if __name__ == "__main__":
    main()
