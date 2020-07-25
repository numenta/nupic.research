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
from pprint import pformat

import wandb
from ray.tune.utils import flatten_dict

__all__ = [
    "log",
    "WandbLogger",
    "WorkerLogger",
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

    It's intended for use in conjunction with the `WorkerLogger` mixin which initializes
    wandb on a specified set of workers. Generally, the ray based logger, `WandbLogger`,
    is intended to log results on the head node, while this function (and the helper
    mixin) serve to log other values on the worker nodes - for instance, grad norms of
    pytroch modules.

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


class WandbLogger(wandb.ray.WandbLogger):
    """
    This subclasses the ray integration from wandb to
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
    # See https://docs.wandb.com/library/log for wandb data-types.
    # Others types may be included later.
    accepted_types = (
        numbers.Number,
        wandb.Image,
        wandb.Histogram,
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

        # Auto format the group to be the name of the trial.
        env_config = self.config["env_config"]
        wandb_config = env_config["wandb"]

        # Find latest run config upon resume.
        resume = wandb_config.get("resume", False)
        if resume and "id" not in wandb_config:
            enable_run_resume(wandb_config)

        # This will invoke `wandb.init(**wandb_config)` and create a new run-directory.
        super()._init()

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


class WorkerLogger(object):
    """
    This class serves an optional mixin for the ImagenetExperiment Class.
    It's purpose is simply to initialize wandb on the worker nodes. This allows
    logging to be done outside of this class by direct calls to wandb. Note that
    the `log` function of this python module is designed with that purpose in mind.

    To keep all logs managed under the same run across the head and worker nodes, try
    using `wandb.util.generate_id()` to get a unique run-id that can be passed to both
    this mixin and the ray based `WandbLogger`.
    """

    def setup_experiment(self, config):
        """
        Init wandb from worker processes if desired. This is useful for debugging
        purposes, such as logging directly from pytorch modules.

        :param config:
            - wandb_args: dict to pass to wandb.init
                - name: name of run
                - project: name of project
                - id: (optional) can generate via wandb.util.generate_id()
                - group: name of group; it's recommended to use this or `id`
                         to keep results together
            - wandb_for_worker_ranks: list of integers denoting the ranks of
                                      processes to init wandb
        """
        super().setup_experiment(config)
        for rank in config.get("wandb_for_worker_ranks", []):

            wandb_args = config.get("wandb_args", {})
            self.logger.info(f"Setting up wandb on rank {rank}")
            self.logger.info(f"wandb_agrs:\n{pformat(wandb_args)}")
            if self.rank == rank:
                wandb.init(**wandb_args)

    def stop_experiment(self):
        """Finalize wandb logging."""
        super().stop_experiment()
        if wandb.run:
            wandb.join()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append(
            "WorkerLogger: setup wandb logging for specified workers")
        eo["stop_experiment"].append(
            "WorkerLogger: finalize wandb logging on worker nodes")
        return eo


# ---------
# Utils
# ---------

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
