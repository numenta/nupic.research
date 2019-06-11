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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tabulate
import pprint
import click
import numpy as np
import pandas as pd
from ray.tune.commands import *

import warnings

warnings.filterwarnings("ignore")


def load(experiment_path):
    """ Load a single experiment into a dataframe """

    experiment_path = os.path.abspath(experiment_path)
    experiment_states = _get_experiment_states(experiment_path, exit_on_fail=True)

    # run once per experiment state
    # columns might differ between experiments
    dataframes = []
    for exp_state, exp_name in experiment_states:
        progress, params = _read_experiment(exp_state, experiment_path)
        dataframes.append(_get_value(progress, params, exp_name))

    # concats all dataframes if there are any and return
    if not dataframes:
        return pd.DataFrame([])
    return pd.concat(dataframes, axis=0, ignore_index=True, sort=False)


def load_many(experiment_paths):
    """ Load several experiments into a single dataframe"""

    dataframes = [load(path) for path in experiment_paths]
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
            paramsFile = os.path.join(experiment_path, exp_dir, "params.json")
            with open(paramsFile) as f:
                params[exp_tag] = json.load(f)

    return progress, params


def _get_value(
    progress,
    params,
    exp_name,
    exp_substring="",
    tags=["test_accuracy", "noise_accuracy", "mean_accuracy"],
    which="max",
):
    """
  For every experiment whose name matches exp_substring, scan the history
  and return the appropriate value associated with tag.
  'which' can be one of the following:
      last: returns the last value
       min: returns the minimum value
       max: returns the maximum value
    median: returns the median value

  Returns a pandas dataframe with two columns containing name and tag value

  Modified to run once per experiment state
  """

    # Collect experiment names that match exp at all
    exps = [e for e in progress if exp_substring in e]

    # empty histories always return None
    columns = ["Experiment Name"]

    # add the columns names for main tags
    for tag in tags:
        columns.append(tag)
        columns.append(tag + "_" + which)
        if which in ["max", "min"]:
            columns.append("epoch_" + str(tag))
    columns.append("epochs")
    columns.append("start_learning_rate")
    columns.append("end_learning_rate")
    columns.append("early_stop")
    columns.append("experiment_file_name")
    columns.extend(["trial_time", "mean_epoch_time"])
    columns.extend(["trial_train_time", "mean_epoch_train_time"])

    # add the remaining variables
    columns.extend(params[exps[0]].keys())

    all_values = []
    for e in exps:
        # values for the experiment name
        values = [e]
        # values for the main tags
        for tag in tags:
            values.append(progress[e][tag].iloc[-1])
            if which == "max":
                values.append(progress[e][tag].max())
                v = progress[e][tag].idxmax()
                values.append(v)
            elif which == "min":
                values.append(progress[e][tag].min())
                values.append(progress[e][tag].idxmin())
            elif which == "median":
                values.append(progress[e][tag].median())
            elif which == "last":
                values.append(progress[e][tag].iloc[-1])
            else:
                raise RuntimeError("Invalid value for which='{}'".format(which))

        # add remaining main tags
        values.append(progress[e]["training_iteration"].iloc[-1])
        values.append(progress[e]["learning_rate"].iloc[0])
        values.append(progress[e]["learning_rate"].iloc[-1])
        # consider early stop if there is a signal and haven't reached last iteration
        if (
            params[e]["iterations"] != progress[e]["training_iteration"].iloc[-1]
            and progress[e]["stop"].iloc[-1]
        ):
            values.append(1)
        else:
            values.append(0)
        values.append(exp_name)
        # store time in minutes
        values.append(progress[e]["epoch_time"].sum() / 60)
        values.append(progress[e]["epoch_time"].mean() / 60)
        values.append(progress[e]["epoch_time_train"].sum() / 60)
        values.append(progress[e]["epoch_time_train"].mean() / 60)

        # remaining values
        for v in params[e].values():
            if isinstance(v, list):
                values.append(np.mean(v))
            else:
                values.append(v)

        all_values.append(values)

    p = pd.DataFrame(all_values, columns=columns)

    return p


def get_checkpoint_file(exp_substring=""):
    """
  For every experiment whose name matches exp_substring, return the
  full path to the checkpoint file. Returns a list of paths.
  """
    # Collect experiment names that match exp at all
    exps = [e for e in progress if exp_substring in e]

    paths = [self.checkpoint_directories[e] for e in exps]

    return paths


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

    experiment_state_paths = list(experiment_state_paths)
    experiment_state_paths.sort()
    experiment_states = []
    for experiment_filename in list(experiment_state_paths):
        with open(experiment_filename) as f:
            experiment_states.append((json.load(f), experiment_filename))

    return experiment_states


def get_parameters(sorted_experiments):
    for i, e in sorted_experiments.iterrows():
        if e["Experiment Name"] in params:
            params = params[e["Experiment Name"]]
            print(params["cnn_percent_on"][0])

    print("test_accuracy")
    for i, e in sorted_experiments.iterrows():
        print(e["test_accuracy"])

    print("noise_accuracy")
    for i, e in sorted_experiments.iterrows():
        print(e["noise_accuracy"])
