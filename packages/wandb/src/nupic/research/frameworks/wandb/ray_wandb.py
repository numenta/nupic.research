# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import ast
import json
import numbers
import os
import warnings
from copy import deepcopy

import wandb
from ray import tune
from ray.tune.utils import flatten_dict

__all__ = [
    "log",
    "WandbLogger",
    "prep_plot_for_wandb",
]

# Find directory of where wandb save its results.
if "WANDB_DIR" in os.environ:
    WANDB_DIR = os.path.join(os.environ["WANDB_DIR"], "wandb")
else:
    WANDB_DIR = None
CONFIG_NAME = "ray_wandb_config.json"


def log(log_dict, commit=False, step=None, sync=True, *args, **kwargs):
    """
    This logs its arguments to wandb only if a run has been initialized.
    """

    if wandb.run:
        wandb.log(log_dict, commit=commit, step=step, sync=sync, *args, **kwargs)


class WandbLogger(tune.logger.Logger):
    """
    This forks the wandb 0.9.7 ray WandbLogger to
        1) make resuming experiments easier
        2) include more supported data types
        3) support logging time-series formatted results

    As a `ray.tune.logger.Logger` this class will process all results returned from
    training and automatically sync them to wandb.

    To use this class, include it in your `tune.run` config under `loggers` and add
    `env_config["wandb"]` to specify wandb params.

    The main options include
        wandb:
            - name: Chosen name of run/experiment
            - project: Name of wandb project to group all related runs
            - group: Extra layer of naming to group runs under a project
            - notes: A multi-line string associated with the run

    All are optional, but name, project, and notes are recommended. For all wandb init
    params, see https://docs.wandb.com/library/init.

    Example usage:
    ```
    # Be sure to set `WANDB_API_KEY` in environment variables.
    from ray.tune.logger import DEFAULT_LOGGERS
    tune.run(
        MyTrianable,
        loggers=list(DEFAULT_LOGGERS) + [WandbLogger],
        config={
            "env_config": {
                "wandb": {
                    "project": "my-project-name",
                    "name": "my-exp-name",
                    "group": "group-of-runs",
                    "notes": "This experiments aims to ..."
                },

                # Optional
                "result_to_time_series_fn":
                MyExperiment.expand_result_to_time_series,
            }
        }
    )
    ```

    The "result_to_time_series_fn" is a function that takes a result and config
    and returns a dictionary of {timestep: result}. If you provide this
    function, you convert from an epoch-based time series to your own
    timestep-based time series, logging multiple timesteps for each epoch.
    """

    # Only the following types are able to be logged through this class.
    # See https://docs.wandb.com/library/log for all wandb data-types.
    accepted_types = (
        numbers.Number,
        wandb.data_types.WBValue,  # Base class for all wandb values
    )

    def _init(self):
        """
        This function runs `wandb.init` with two key extra steps:
            1) `group` is automatically assigned to the date-time if not already given
            2) The config passed to `wandb.init` is saved. This allows `log` (from this
               module) to make an identical call to `wandb.init`. While The former init
               gets called outside of the ray process, the latter typically does not.
               Thus, by saving the wandb-config, we can associate calls to `log` to the
               same `group` associated to this logger.
        """
        self._config = None

        # Auto format the group to be the name of the trial.
        env_config = self.config["env_config"]
        wandb_config = env_config["wandb"]

        # Find latest run config upon resume.
        resume = wandb_config.get("resume", False)
        if resume and "id" not in wandb_config:
            enable_run_resume(wandb_config)

        # This will create a new run-directory.
        wandb.init(**self.config.get("env_config", {}).get("wandb", {}))

        # Get result_to_time_series_fn.
        experiment_class = self.config.get("experiment_class", None)
        self.result_to_time_series_fn = None

        if "result_to_time_series_fn" in env_config:
            self.result_to_time_series_fn = env_config["result_to_time_series_fn"]
        elif hasattr(experiment_class, "expand_result_to_time_series"):
            self.result_to_time_series_fn = (
                experiment_class.expand_result_to_time_series
            )

    def on_result(self, result):
        """
        The following is copied from the parent class; however, non-serializable
        config values are saved as the repr's so that they are all yaml
        serializable. See for details:
            - https://github.com/wandb/client/issues/586
        """

        config = deepcopy(result.get("config"))
        if config and self._config is None:
            for k in config.keys():
                if wandb.config.get(k) is None:
                    s = repr(config[k])
                    try:
                        ast.literal_eval(s)
                        wandb.config[k] = config[k]
                    except (ValueError, SyntaxError):
                        # Non-serializable
                        wandb.config[k] = s
            self._config = config

        tmp = result.copy()
        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]

        if self.result_to_time_series_fn is not None:
            assert self._config is not None
            time_series_dict = self.result_to_time_series_fn(tmp, self._config)
            for t, d in sorted(time_series_dict.items(), key=lambda x: x[0]):
                metrics = {}
                for key, value in flatten_dict(d, delimiter="/").items():
                    if not isinstance(value, self.accepted_types):
                        continue
                    metrics[key] = value
                wandb.log(metrics, step=t)
        else:
            metrics = {}
            for key, value in flatten_dict(tmp, delimiter="/").items():
                if not isinstance(value, self.accepted_types):
                    continue
                metrics[key] = value
            wandb.log(metrics)

    def close(self):
        wandb.join()


class PrepPlotForWandb:
    """
    This mixin ensures all plots can be logged to wandb without error. Ray typically
    tries to deepcopy the results dict which throws an error since this is not
    implemented for plots by matplotlib. This is avoided by first wrapping the plots
    with wandb.Image before sending them to Ray which logs them through the WandbLogger.
    """

    def run_epoch(self):
        """Wrap plots with wandb.Image"""
        results = super().run_epoch()

        wandb_plots = {}
        for name, value in results.items():
            if is_matplotlib_plot(value):
                wandb_plots[name] = wandb.Image(value)

        results.update(wandb_plots)
        return results

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["run_epoch"].append(
            "PrepPlotForWandb: Wrap plots with wandb.Image")
        return eo


# ---------
# Utils
# ---------


def is_matplotlib_plot(value):
    typename = wandb.util.get_full_typename(value)
    return wandb.util.is_matplotlib_typename(typename)


def prep_plot_for_wandb(plot_func):
    """
    This wraps a plotting function to alter it's return value to be of type wandb.Image.
    This way, the plot can be logged through ray, specifically the ray WandbLogger,
    without error. Ray typically tries to deepcopy all logged objects; however, plots
    cannot be deepcopied.

    :param plot_func: callable with arbitrary arguments that returns a matplotlib
                      figure, axes object, or anything related.
    """
    def plot_and_make_wandb_image(*args, **kwargs):
        plot = plot_func(*args, **kwargs)
        if is_matplotlib_plot(plot):
            plot = wandb.Image(plot)
        else:
            warnings.warn(f"Unable to convert object of type {type(plot)}"
                          " to `wandb.Image`.")
        return plot

    return plot_and_make_wandb_image


def enable_run_resume(wandb_config):
    """
    Finds and sets latest wandb run id to resume the corresponding run.
    """
    name = wandb_config.get("name", None)
    run_id = wandb_config.get("id", None)
    if name and not run_id:
        run_id = get_latest_run_id(name=name) or None
        if run_id is None:
            warnings.warn(
                "Couldn't find latest wandb run-id to resume."
                "Ensure `WANDB_DIR` environment variable is set.")

    wandb_config.update(id=run_id, resume=True)


def get_latest_run_id(name=None):
    """
    Gets the config of the latest wandb run.

    :param name: (optional) name of run; filters runs so they must match the name given
    """

    latest_run_dir = get_latest_run_dir(name=name)
    if latest_run_dir is None:
        return None

    run_id = latest_run_dir.split("-")[-1] or None  # None if empty string
    return run_id


def get_latest_run_dir(name=None):
    """
    Gets the directory of where the latest wandb run is saved.

    :param name: (optional) name of run; filters runs so they must match the name given
    """

    if WANDB_DIR is None:
        return None

    all_subdirs = []
    for d in os.listdir(WANDB_DIR):

        # Make sure run directory exists.
        d_full = os.path.join(WANDB_DIR, d)
        if not os.path.isdir(d_full):
            continue

        # Validate name of run when specified.
        run_metadata_path = os.path.join(d_full, "wandb-metadata.json")
        if name and os.path.isfile(run_metadata_path):
            with open(run_metadata_path, "r") as f:
                try:
                    run_metadata = json.load(f)
                except json.JSONDecodeError:
                    run_metadata = {}

            d_name = run_metadata.get("name", False)
            if d_name and d_name == name:
                all_subdirs.append(d_full)

        # If name is not given, add to list of run directories by default.
        elif name is None:
            all_subdirs.append(d_full)

    # Find latest run directory chronologically.
    latest_run_dir = max(all_subdirs, key=os.path.getmtime)
    return latest_run_dir
