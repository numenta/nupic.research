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
import os
import re
from itertools import groupby
from pathlib import Path

import pandas as pd

from nupic.research.support import load_ray_tune_experiments, parse_config


# Select a unique tag for each parameter combination, ignoring seed value
# Used to group multiple random seeds of the same configuration for computing results.
def key_func(x):
    s = re.split("[,]", re.sub(",|\\d+_|seed=\\d+", "", x["experiment_tag"]))
    if len(s[0]) == 0:
        return [" "]
    return s


def parse_one_experiment(exp, states, df):
    """
    Parse the trials in one experiment and append data to the given dataframe.
    :param exp:
    :param states:
    :param config_table:

    :return: a new dataframe with the results (the original one is not modified)
    """
    df_entries = []
    for experiment_state in states:
        # Go through all checkpoints in the experiment
        all_trials = experiment_state["checkpoints"]

        # Group trials based on their parameter combinations (represented by tag)
        parameter_groups = {
            k[0]: list(v)
            for k, v in groupby(sorted(all_trials, key=key_func), key=key_func)
        }

        for tag in parameter_groups:
            trial_checkpoints = parameter_groups[tag]

            try:
                for _, trial_checkpoint in enumerate(trial_checkpoints):
                    results = trial_checkpoint["results"]
                    if results is None:
                        continue

                    # For each checkpoint select the epoch with the best accuracy as
                    # the best epoch
                    best_result = max(results, key=lambda x: x["mean_accuracy"])

                    # Get network params and weight sparsities for this parameter group
                    net_params = trial_checkpoint["config"]
                    l1c = net_params["cnn_out_channels"][0]
                    l2c = net_params["cnn_out_channels"][1]
                    l3n = net_params["linear_n"][0]
                    l1w = net_params["cnn_weight_sparsity"][0]
                    if l1w > 0.0:
                        l1w = 1.0 - l1w
                    l2w = net_params["cnn_weight_sparsity"][1]
                    if l2w > 0.0:
                        l2w = 1.0 - l2w
                    l3w = net_params["weight_sparsity"][0]
                    if l3w > 0.0:
                        l3w = 1.0 - l3w
                    l2d = l1c * 25  # Each L2 kernel is 5x5
                    l3d = l2c * 25  # The last CNN layer after maxpool is 5x5

                    # Do we have any activation sparsity
                    activation_sparsity = 0
                    if ((net_params["cnn_percent_on"][1] < 1.0) or
                        (net_params["linear_percent_on"][0] < 1.0)):
                        activation_sparsity = 1

                    for iteration, r in enumerate(results):
                        df_entries.append([
                            exp,
                            l1c, l2c, l3n,
                            l1w, l2w, l3w,
                            activation_sparsity,
                            r["non_zero_parameters"], r["mean_accuracy"],
                            iteration, best_result["mean_accuracy"],
                            l2d, l3d, results[0]["config"]["seed"],
                            "{} {}".format(exp, tag)
                        ])

            except Exception:
                print("Problem with checkpoint group" + tag + " in " + exp
                      + " ...skipping")
                continue

    # Create new dataframe from the entries with same dimensions as df
    df2 = pd.DataFrame(df_entries, columns=df.columns)
    return df.append(df2)


def parse_results(config_filename, experiments):
    """
    Parse the results for each specified experiment in one cfg file. Creates a
    dataframe containing one row per iteration for every trial for every
    network configuration in every experiment.

    The dataframe is saved to config_filename.pkl The raw results are also saved in a
    .csv file named config_filename.csv.

    :param config_filename: the cfg filename
    :param experiments: a list of experiment names from the cfg file

    :return: a dataframe containing raw results
    """

    # The results table
    columns = ["Experiment name", "L1 channels", "L2 channels", "L3 N",
               "L1 Wt sparsity", "L2 Wt sparsity", "L3 Wt sparsity",
               "Activation sparsity",
               "Non-zero params", "Accuracy", "Iteration", "Best accuracy",
               "L2 dimensionality", "L3 dimensionality", "Seed", "ID"
               ]
    df = pd.DataFrame(columns=columns)

    # Load and parse experiment configurations
    with open(config_filename, "r") as config_file:
        configs = parse_config(config_file, experiments, globals_param=globals())

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
            print("Could not locate experiment state for " + exp + " ...skipping")
            continue

        df = parse_one_experiment(exp, states, df)

    df.to_csv(os.path.splitext(config_file.name)[0] + ".csv")
    df.to_pickle(os.path.splitext(config_file.name)[0] + ".pkl")
    return df


if __name__ == "__main__":
    parse_results(config_filename="dense_scaling_baselines.cfg",
                  experiments=["Dense_Baselines"])
