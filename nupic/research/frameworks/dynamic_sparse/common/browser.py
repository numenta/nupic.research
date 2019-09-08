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

"""
Module for browsing and manipulating experiment results directories created
by Ray Tune.

Converted to a module
"""

from __future__ import absolute_import, division, print_function

import glob
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from ray.tune.commands import flatten_dict

warnings.filterwarnings("ignore")


def load(experiment_path, metrics=None):
    """Load a single experiment into a dataframe"""
    experiment_path = os.path.abspath(experiment_path)
    experiment_states = _get_experiment_states(experiment_path, exit_on_fail=True)

    # run once per experiment state
    # columns might differ between experiments
    dataframes = []
    for exp_state, exp_name in experiment_states:
        progress, params = _read_experiment(exp_state, experiment_path)
        dataframes.append(_get_value(progress, params, exp_name, metrics))

    # concats all dataframes if there are any and return
    if not dataframes:
        return pd.DataFrame([])
    return pd.concat(dataframes, axis=0, ignore_index=True, sort=False)


def load_many(experiment_paths, metrics=None):
    """Load several experiments into a single dataframe"""
    dataframes = [load(path, metrics) for path in experiment_paths]
    return pd.concat(dataframes, axis=0, ignore_index=True, sort=False)


def _read_experiment(experiment_state, experiment_path):
    checkpoint_dicts = experiment_state["checkpoints"]
    checkpoint_dicts = [flatten_dict(g) for g in checkpoint_dicts]

    progress = {}
    params = {}
    # TODO: no real use for exp_directories outside this function, why get it?
    exp_directories = {}
    for exp in checkpoint_dicts:
        if exp.get("logdir", None) is None:
            continue
        exp_dir = os.path.basename(exp["logdir"])
        exp_tag = exp["experiment_tag"]
        csv = os.path.join(experiment_path, exp_dir, "progress.csv")
        # check if file size is > 0 before proceeding
        if os.stat(csv).st_size:
            progress[exp_tag] = pd.read_csv(csv)
            exp_directories[exp_tag] = os.path.abspath(
                os.path.join(experiment_path, exp_dir)
            )

            # Read in the configs for this experiment
            params_file = os.path.join(experiment_path, exp_dir, "params.json")
            with open(params_file) as f:
                params[exp_tag] = json.load(f)

    return progress, params


def _get_value(progress, params, exp_name, performance_metrics=None, exp_substring=""):
    """
    For every experiment whose name matches exp_substring, scan the history
    and return the appropriate value associated with tag.

    Allow for custom performance metrics, such as ["test_accuracy", mean_accuracy"]
    For performance metrics, will collect max and min (and the respective epoch)
    along with median and last values

    Can be modified to add more custom metrics based on available params

    Returns a pandas dataframe with two columns containing name and tag value

    Modified to run once per experiment state
    """
    # Collect experiment names that match exp at all
    exps = [e for e in progress if exp_substring in e]

    # try to automatically determine what are the performance metrics, if not given
    if performance_metrics is None:
        performance_metrics = [m for m in progress[exps[0]].keys() if "acc" in m]

    # populate stats
    stats = defaultdict(list)
    for e in exps:
        # add relevant progress metrics
        stats["Experiment Name"].append(e)
        for m in performance_metrics:
            # max
            stats[m + "_max"].append(progress[e][m].max())
            stats[m + "_max_epoch"].append(progress[e][m].idxmax())
            # min
            stats[m + "_min"].append(progress[e][m].min())
            stats[m + "_min_epoch"].append(progress[e][m].idxmin())
            # others
            stats[m + "_median"].append(progress[e][m].median())
            stats[m + "_last"].append(progress[e][m].iloc[-1])

        # remaining custom tags - specific
        stats["epochs"].append(progress[e]["training_iteration"].iloc[-1])
        stats["experiment_file_name"].append(exp_name)
        stats["trial_time"].append(progress[e]["time_this_iter_s"].sum() / 60)
        stats["mean_epoch_time"].append(progress[e]["time_this_iter_s"].mean() / 60)

        # removed - couldn't find related in current dataset
        # stats["start_learning_rate"].append(progress[e]["learning_rate"].iloc[0])
        # stats["end_learning_rate"].append(progress[e]["learning_rate"].iloc[-1])
        # stats["trial_train_time"].append(progress[e]["time_this_iter_s"].sum() / 60)
        # stats["mean_epoch_train_time"].append(progress[e]["time_this_iter_s"].mean()/60)

        # early stop
        # if (
        #     params[e]["iterations"] != progress[e]["training_iteration"].iloc[-1]
        #     and progress[e]["stop"].iloc[-1]
        # ):
        #     stats["early_stop"].append(1)
        # else:
        #     stats["early_stop"].append(0)

        # add all remaining params, for easy aggregations
        for k, v in params[e].items():
            if isinstance(v, list):
                stats[k].append(np.mean(v))
            else:
                stats[k].append(v)

    return pd.DataFrame(stats)


def _get_experiment_states(experiment_path, exit_on_fail=False):
    """
    Return every experiment state JSON file in the path as a list of dicts.
    The list is sorted such that newer experiments appear later.
    """
    experiment_path = os.path.expanduser(experiment_path)
    experiment_state_paths = glob.glob(
        os.path.join(experiment_path, "experiment_state*.json")
    )

    if not experiment_state_paths:
        print("No experiment state found for experiment {}".format(experiment_path))
        return []

    experiment_state_paths = sorted(experiment_state_paths)
    experiment_states = []
    for experiment_filename in list(experiment_state_paths):
        with open(experiment_filename) as f:
            experiment_states.append((json.load(f), experiment_filename))

    return experiment_states
