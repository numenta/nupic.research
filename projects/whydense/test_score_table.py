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

from nupic.research.support import load_ray_tune_experiments, parse_config


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
    type=click.Choice(choices=["grid", "latex_raw"]),
    show_default=True,
    default="grid",
)
def main(config, experiments, tablefmt):

    # The table we use in the paper
    test_scores_table = [["Network", "Test Score", "Noise Score",
                          "Params"]]

    # A more detailed table
    test_scores_table_long = [["Network", "Test Score", "Noise Score", "Noise Accuracy",
                               "Total Entropy", "Nonzero Parameters", "Num Trials",
                               "Session"]]

    # Load and parse experiment configurations
    configs = parse_config(config, experiments, globals_param=globals())

    # Use the appropriate plus/minus sign for latex
    if tablefmt == "grid":
        pm = "±"
    else:
        pm = "$\\pm$"

    # Select tags ignoring seed value
    def key_func(x):
        s = re.split("[,]", re.sub(",|\\d+_|seed=\\d+", "", x["experiment_tag"]))
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
        try:
            states = load_ray_tune_experiments(
                experiment_path=experiment_path, load_results=True
            )

        except RuntimeError:
            # print("Could not locate experiment state for " + exp + " ...skipping")
            continue

        for experiment_state in states:
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
                noise_accuracies = np.zeros(num_exps)
                noise_samples = np.zeros(num_exps)
                nonzero_params = np.zeros(num_exps)
                entropies = np.zeros(num_exps)

                try:
                    for i, checkpoint in enumerate(checkpoints):
                        results = checkpoint["results"]
                        if results is None:
                            continue

                        # For each checkpoint select the epoch with the best accuracy as
                        # the best epoch
                        best_result = max(results, key=lambda x: x["mean_accuracy"])
                        test_scores[i] = best_result["mean_accuracy"]
                        entropies[i] = best_result["entropy"]
                        # print("best result:", best_result)

                        # Load noise score
                        logdir = os.path.join(
                            experiment_path, os.path.basename(checkpoint["logdir"])
                        )
                        filename = os.path.join(logdir, "noise.json")
                        if os.path.exists(filename):
                            with open(filename, "r") as f:
                                noise = json.load(f)

                            noise_scores[i] = sum(x["total_correct"]
                                                  for x in list(noise.values()))
                            noise_samples[i] = sum(
                                x["total_samples"] for x in list(noise.values()))
                        else:
                            print("No noise file for " + experiment_path
                                  + " ...skipping")
                            continue

                        noise_accuracies[i] = (
                            float(100.0 * noise_scores[i]) / noise_samples[i])
                        nonzero_params[i] = max(x["non_zero_parameters"]
                                                for x in list(noise.values()))
                except Exception:
                    print("Problem with checkpoint group" + tag + " in " + exp
                          + " ...skipping")
                    continue

                test_score = "{0:.2f} {1:} {2:.2f}".format(
                    test_scores.mean(), pm, test_scores.std()
                )
                entropy = "{0:.2f} {1:} {2:.2f}".format(
                    entropies.mean(), pm, entropies.std()
                )
                noise_score = "{0:,.0f} {1:}  {2:.2f}".format(
                    noise_scores.mean(), pm, noise_scores.std()
                )
                noise_accuracy = "{0:,.2f} {1:}  {2:.2f}".format(
                    noise_accuracies.mean(), pm, noise_accuracies.std()
                )
                nonzero = "{0:,.0f}".format(nonzero_params.mean())
                test_scores_table.append(
                    ["{} {}".format(exp, tag), test_score, noise_accuracy,
                     nonzero]
                )
                test_scores_table_long.append(
                    ["{} {}".format(exp, tag), test_score, noise_score, noise_accuracy,
                     entropy, nonzero, num_exps,
                     experiment_state["runner_data"]["_session_str"]]
                )

    print()
    print(tabulate(test_scores_table, headers="firstrow", tablefmt=tablefmt))
    print()
    print(tabulate(test_scores_table_long, headers="firstrow", tablefmt=tablefmt))


if __name__ == "__main__":
    main()
