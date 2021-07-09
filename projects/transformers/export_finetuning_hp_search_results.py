#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
"""
Code for analyzing hyperparameter search runs conducted with Ray Tune.
Adapted from nupic.research/nupic/research/archive/dynamic_sparse/browser.
"""


from __future__ import absolute_import, division, print_function

import codecs
import copy
import glob
import json
import numbers
import os
import pickle
import sys
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as ss
from sklearn import linear_model

from finetuning_constants import (
    ALL_REPORTING_METRICS,
    GLUE_NAMES_PER_TASK,
    REPORTING_METRICS_PER_TASK,
    TASK_NAMES,
)

warnings.filterwarnings("ignore")

# ---------
# Panda Utils (old / not currently used)
# ---------


# helper functions
def mean_and_std(s):
    return "{:.3f} ± {:.3f}".format(s.mean(), s.std())


def round_mean(s):
    return "{:.0f}".format(round(s.mean()))


stats = ["min", "max", "mean", "std"]


def agg(df, columns, metric="eval_loss", filter_by=None, decimals=3):
    if filter_by is None:
        return (
            df.groupby(columns).agg(
                {f"{metric}_epoch": round_mean, metric: stats, "seed": ["count"]}
            )
        ).round(decimals)
    else:
        return (
            df[filter_by]
            .groupby(columns)
            .agg({f"{metric}_epoch": round_mean, metric: stats, "seed": ["count"]})
        ).round(decimals)


# ---------
# Utils (old / barely used)
# ---------


def flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def unpickle_object(x):
    """
    Decodes pickled objects that are encoded in `base64`.

    Example:
    ```
        a = np.array([1, 2, 3])
        pickled = codecs.encode(pickle.dumps(a), "base64").decode()
        unpickled = numpy_from_pickled(pickled)
        assert np.allclose(a, unpickled)
    ```

    """
    x = x.encode()
    x = codecs.decode(x, "base64")
    x = pickle.loads(x)
    return x


def unpickle_within_series(series):
    """
    Helps unpickle items within pandas series.
    """
    if isinstance(series, pd.core.series.Series):
        series = series.apply(unpickle_object)
    else:
        series = unpickle_object(series)
    return series


def unpickle_within_dataframe(df, conditions):
    """
    Helps unpickle items within Dataframe.
    """
    df_copy = df.copy()
    for col in df.columns:
        if any([cond(col) for cond in conditions]):
            df_copy[col] = df[col].apply(unpickle_within_series)
    return df_copy


# ---------------------------
# Functions for aggregating data from hyperparameter tuning experiments
# ---------------------------


def load(experiment_path):
    """Load a single experiment into a dataframe"""
    experiment_path = os.path.expanduser(experiment_path)
    experiment_states = _get_experiment_states(experiment_path, exit_on_fail=True)

    # run once per experiment state
    # columns might differ between experiments
    dataframes = []
    for exp_state, exp_name in experiment_states:
        progress, params = _read_experiment(exp_state, experiment_path)
        if len(progress) != 0:
            dataframes.append(
                _get_value(progress, params, exp_name)
            )

    # concats all dataframes if there are any and return
    if not dataframes:
        return pd.DataFrame([])
    return pd.concat(dataframes, axis=0, ignore_index=True, sort=False)


def get_all_subdirs(experiment_path):
    """
    This assumes a directory structure as follows:

        name_of_model
            - task 1
                - experiment 1
                - experiment 2
                - ...
            - task 2
            - ...
        name_of_next_model
        ...
    
    experiment path points towards a single model
    """

    experiment_path = os.path.expanduser(experiment_path)
    files = os.listdir(experiment_path)

    subdirs = []
    for f in files:
        file_path = os.path.join(experiment_path, f)
        if os.path.isdir(file_path):
            subdirs.append(file_path)

    experiment_state_paths = glob.glob(
        os.path.join(experiment_path, "experiment_state*.json")
    )

    has_exp = False
    if experiment_state_paths:
        has_exp = True

    return subdirs, has_exp


def check_for_tasks(base_path):

    results = []
    task_2_path = {}
    files = os.listdir(base_path)
    for file in files:
        file_path = os.path.join(base_path, file)
        if os.path.isdir(file_path):
            if file in TASK_NAMES:
                task_2_path[file] = file_path
                results.append(file_path)
    
    return results, task_2_path


def load_from_base(base_path):
    """Look for all experimental subdirectories and load each into a dataframe"""

    # gather tasks for which we have hp searches, for the model specified in
    # base_path
    results, task_2_path = check_for_tasks(base_path)
    task_2_subdirs = {}
    for task in task_2_path.keys():
        task_path = task_2_path[task]
        experiment_paths, has_exp = get_all_subdirs(task_path)
        task_2_subdirs[task] = experiment_paths

    task_2_dfs = {}
    for task in task_2_subdirs.keys():
        task_2_dfs[task] = []
        for subdir in task_2_subdirs[task]:
            df = load(subdir)
            task_2_dfs[task].append(df)
        task_2_dfs[task] = pd.concat(task_2_dfs[task])

    return task_2_dfs


def _read_experiment(experiment_state, experiment_path):

    checkpoint_dicts = experiment_state["checkpoints"]
    checkpoint_dicts = [flatten_dict(json.loads(g)) for g in checkpoint_dicts]

    progress = {}
    params = {}

    for exp in checkpoint_dicts:
        if exp.get("logdir", None) is None:
            continue
        exp_dir = os.path.basename(exp["logdir"])
        exp_tag = exp["experiment_tag"]
        csv = os.path.join(experiment_path, exp_dir, "progress.csv")

        # check if file exists and size > 0
        if os.path.exists(csv):
            if os.stat(csv).st_size:
                progress[exp_tag] = pd.read_csv(csv)

            # Read in the configs for this experiment
            params_file = os.path.join(experiment_path, exp_dir, "params.json")
            with open(params_file) as f:
                params[exp_tag] = json.load(f)

    return progress, params


def _get_value(progress, params, exp_name):
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

    # Collect experiment names
    exps = [e for e in progress]

    # Use all performance metrics from ALL_REPORTING_METRICS, if present
    performance_metrics = []
    columns = progress[exps[0]].keys()
    for metric in ALL_REPORTING_METRICS:
        if metric in columns:
            performance_metrics.append(metric)

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


        # add all remaining params, for easy aggregations
        for k, v in params[e].items():
            # TODO: fix this hard coded check, added as temporary fix
            if k != "epochs":
                # take the mean if a list of numbers
                if isinstance(v, list) and all(
                    [isinstance(n, numbers.Number) for n in v]
                ):
                    stats[k].append(np.mean(v))
                # concatenate if a list of strings
                elif isinstance(v, list):
                    v = "-".join([str(v_i) for v_i in v])
                    stats[k].append(v)
                # iterate through dictionaries to get keys inside
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        stats["_".join([k, k2])].append(v2)
                # otherwise append the value as it is
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


def save_agg_results(task_2_df, experiment_path):

    for task in task_2_df.keys():
        save_path = os.path.join(experiment_path, task)
        assert os.path.exists(save_path)
        save_file = os.path.join(save_path, f"{task}.csv")
        print(f"Saving {task} results to {save_file}")
        task_2_df[task].to_csv(save_file)


# ---------------------------
# Analysis (plots, regression, etc.)
# ---------------------------


def plot_categorical(df, column_name):
    pass


def plot_categorical_bar(df, column_name):
    pass


def lin_regress_1d_metric_onto_hps(df, metric, column_names=None, task_name=None):

    if column_names is None:
        column_names = ['learning_rate', 'max_steps', 'warmup_ratio']

    hp_regs = {}
    y = df[metric].values
    X = df[column_names].values
    for col in column_names:
        x = df[col].values
        result = ss.linregress(x, y)
        print(result)
        hp_regs[col] = result

    return hp_regs, X, y


def plot_hp_regs(X, y, hp_regs, task_name=None, **subplot_kwargs):

    task_name = task_name if task_name else ""

    fig, ax = plt.subplots(1, len(hp_regs), **subplot_kwargs)
    for idx, param in enumerate(hp_regs):

        reg = hp_regs[param]
        ax[idx].plot(X[:,idx], y, "b.", label="data")
        ax[idx].plot(X[:,idx], reg.intercept + reg.slope * X[:,idx],
            'r', linestyle='dashed', label="lin reg"
        )
        ax[idx].set_title(task_name + "_" + param)

    return fig, ax


def reg_and_plot(df, metric, column_names=None, task_name=None, **kwargs):

    hp_regs, X, y = lin_regress_1d_metric_onto_hps(df, metric, column_names)
    fig, ax = plot_hp_regs(X, y, hp_regs, task_name, **kwargs)

    return hp_regs, X, y, fig, ax











if __name__ == "__main__":
    experiment_path = str(sys.argv[1])
    task_2_df = load_from_base(experiment_path)
    save_agg_results(task_2_df, experiment_path)