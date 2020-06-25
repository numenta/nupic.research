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
from ray.tune.utils import flatten_dict

__all__ = [
    "log",
    "WandbLogger",
]

# Find directory of where wandb save its results.
CONFIG_NAME = "wandb_config.json"
if "WANDB_DIR" in os.environ:
    WANDB_DIR = os.path.join(os.environ["WANDB_DIR"], "wandb")
else:
    WANDB_DIR = None


def log(log_dict, commit=False, step=None, sync=True, *args, **kwargs):
    """
    This logs its arguments to wandb only if a run has been initialized.

    It's intended for use with the `WandbLogger` class of this module which manages
    runs. In the multiprocessing setting, only the rank 0 process will have logging
    enabled to call `wandb.log` through this function. In general, the user should
    be careful to ensure wandb has been initialized whenever desired. Otherwise, this
    function will be mute.

    Example:
    ```
    plt = plot(...)
    log({"accuracy": 0.9, "epoch": 5, "my_plot": plt})
    ```

    See `wandb.log` usage for details:
        - https://docs.wandb.com/library/log
        - https://docs.wandb.com/library/log#incremental-logging

    Note: `commit` defaults to False here instead of True. This means the call to
          `wandb.log` will not advance the step count. See the link on incremental
          logging.

    :param row: A dict of serializable python objects i.e str: ints, floats, Tensors,
                dicts, or wandb.data_types
    :param commit: Persist a set of metrics, if false just update the existing dict
                   (defaults to true if step is not specified)
    :param step: The global step in processing. This persists any non-committed earlier
                 steps but defaults to not committing the specified step
    :param sync: If set to False, process calls to log in a separate thread
    """

    if wandb.run:
        wandb.log(log_dict, commit=commit, step=step, sync=sync, *args, **kwargs)


class WandbLogger(object):
    """
    Mixin class for experiment logging through wandb.
    """

    # Only the following types are able to be logged through this class.
    # See https://docs.wandb.com/library/log for wandb data-types.
    # Others types may be included later.
    accepted_types = (
        numbers.Number,
        wandb.Image,
        wandb.Histogram,
    )

    def setup_experiment(self, config):
        """
        This function runs `wandb.init(**config["wandb_args"])` and
            1) logs the experiment config to wandb
            1) (as needed) resumes logging of unfinished experiments
        As of now, logging is only initialized on the rank 0 process.

        :param config: dictionary of wandb related settings and experiment variables

            - wandb_args: dict of keyword arguments to pass to `wandb.init`
                - name: name of experiment
                - project: name of project
                - resume: True or False to resume last run. This will automatically
                          find the latest experiment run with `name`.
                - <other args>: See https://docs.wandb.com/library/init for full
                                list of arguments.

            - <experiment variable(s)>: optional items to be logged to wandb as the
                                        experiment's configuration variables

        Other Settings:

        1 ) Subclasses may define a "expand_result_to_time_series" class function. This
        is a function that takes a result and config and returns a dictionary of
        {timestep: result}. If you provide this function, you convert from an
        epoch-based time series to your own timestep-based time series, logging multiple
        timesteps for each epoch.

        2) To enable resuming previously unfinished experiments, set the `WANDB_DIR`
        environment variable or include the desired run `id` in the config. If the
        run id is not given, this class will look through the wandb-directory and find
        the latest run by the same experiment name.
        """

        super().setup_experiment(config)
        if self.rank != 0:
            self._log_results = False
            return  # only setup wandb on the first (zero-ith) process
        else:
            self._log_results = True

        # Get `wandb.init` arguments.
        config = deepcopy(config)
        wandb_config = config.pop("wandb_args")

        # Find latest run config upon resume.
        resume = wandb_config.get("resume", False)
        if resume:
            self.enable_run_resume(wandb_config)

        # Initialize the wandb run - this will create a new run directory unless
        # `wandb_config["resume"]=True`.
        wandb.init(**wandb_config)

        # Log experiment config - must be called after `wandb.init`.
        self.log_config(config)
        self._config = config

        # Enable results -> time-series; This may be overridden by other mixins.
        if hasattr(self, "expand_result_to_time_series"):
            self.result_to_time_series_fn = self.expand_result_to_time_series

    def enable_run_resume(self, wandb_config):
        """
        Finds and sets latest wandb run id to resume the corresponding run.
        """
        name = wandb_config.get("name", None)
        run_id = wandb_config.get("id", None)
        if name and not run_id:
            run_id = get_latest_run_id(name=name) or ""
            if run_id is None:
                warnings.warn(
                    "Couldn't find latest wandb run-id to resume."
                    "Ensure `WANDB_DIR` environment variable is set.")

        wandb_config.update(id=run_id, resume=True)

    def log_config(self, experiment_config):
        """
        Logs experiment's config to wandb.
        """

        config = deepcopy(experiment_config)
        for k in config.keys():
            if wandb.config.get(k) is None:
                s = repr(config[k])
                try:
                    ast.literal_eval(s)
                    wandb.config[k] = config[k]
                except (ValueError, SyntaxError):
                    # Non-serializable
                    wandb.config[k] = s

    def log_result(self, result):
        """
        Logs results dict to dictionary. Value types must be instances
        of types as listed in `accepted_types`.
        """
        super().log_result(result)

        # Logging should only occur in the zero-ith process.
        if not self._log_results:
            return

        if self.result_to_time_series_fn is not None:
            assert self._config is not None
            time_series_dict = self.result_to_time_series_fn(result, self._config)
            for t, d in sorted(time_series_dict.items(), key=lambda x: x[0]):
                metrics = {}
                for key, value in flatten_dict(d, delimiter="/").items():
                    if not isinstance(value, self.accepted_types):
                        continue
                    metrics[key] = value
                wandb.log(metrics, step=t)
        else:
            metrics = {}
            for key, value in flatten_dict(result, delimiter="/").items():
                if not isinstance(value, self.accepted_types):
                    continue
                metrics[key] = value
            wandb.log(metrics)

    def stop_experiment(self):
        """Finalize wandb logging."""
        super().stop_experiment()

        # Logging should only occur in the zero-ith process.
        if wandb.run:
            wandb.join()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("WandbLogger: setup wandb logging for rank 0")
        eo["log_result"].append("WandbLogger: log results from rank 0 process")
        eo["stop_experiment"].append("WandbLogger: finalize wandb logging")
        return eo


# ---------
# Utils
# ---------

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
