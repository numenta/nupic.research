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

import json
import numbers
import os
import warnings
from copy import deepcopy
from datetime import datetime

from ray.tune.utils import flatten_dict

import wandb

__all__ = [
    "log",
    "WandbLogger",
    "auto_init",
]

# Find directory of where wandb save it's results.
if "WANDB_DIR" in os.environ:
    WANDB_DIR = os.path.join(os.environ["WANDB_DIR"], "wandb")
else:
    WANDB_DIR = None
CONFIG_NAME = "ray_wandb_config.json"


def auto_init():
    """Auto init to last run."""
    try:
        latest_config = get_latest_run_config()
        if latest_config:
            wandb.init(**latest_config)
        else:
            warnings.warn("Unable to load and init config from last run.")

        # Save the config to the latest run directory.
        latest_run_dir = get_latest_run_dir()
        if latest_run_dir and latest_config:
            save_wandb_config(latest_config, run_dir=latest_run_dir)

    except FileNotFoundError:
        print("Unable to init wandb from last run.")


def log(log_dict, commit=False, step=None, sync=True, *args, **kwargs):
    """
    This logs its arguments to Weights and Biases. Use this function along with the
    `WandbLogger` which saves the wandb-configuration and associated run. Whenever the
    run is unknown, `log` will automatically default to latest one - thereby defaulting
    to the same run as the logger.

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

    if wandb.run is None:
        auto_init()

    if wandb.run:
        wandb.log(log_dict, commit=commit, step=step, sync=sync, *args, **kwargs)


class WandbLogger(wandb.ray.WandbLogger):
    """
    This subclasses the ray integration from wandb, to achieve automatic grouping of
    logs. For information on grouping and the parent class, see:
        - https://docs.wandb.com/library/integrations/ray-tune
        - https://docs.wandb.com/library/advanced/grouping

    To use this class, pass it under `loggers` and add `env_config["wandb"]` to the
    ray.tune config. For example,

    ```
    from ray.tune.logger import DEFAULT_LOGGERS
    tune.run(
        MyTrianable,
        loggers=DEFAULT_LOGGERS + [WandbLogger],
        config={
            "monitor": True,
            "env_config": {
                "wandb": {
                    "project": "my-project-name",
                    "monitor_gym": True
                }
            }
        }
    )
    ```

    Note, as a `ray.tune.logger.Logger` this class will process all results returned
    from training. However, as of now, only numbers get synced to wandb (e.g. {acc: 1}).
    For logging non-numerical values (such as plots), invoke `log` as defined within
    this module.
    """

    accepted_types = (
        numbers.Number,
        # wandb.Image,  # this seems to be have issues.
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
        wandb_config = self.config["env_config"]["wandb"]
        if "group" not in wandb_config:
            wandb_config["group"] = (
                "group_" + datetime.now().strftime("%Y-%m-%dT%H_%M_%S_%f%z")
            )

        # This will invoke `wandb.init(**wandb_config)` and create a new run-directory.
        super()._init()

        # Add unique run id to the config.
        wandb_config["id"] = wandb.run.id

        # Save the config to the latest run directory.
        latest_run = get_latest_run_dir()
        if latest_run:
            save_wandb_config(wandb_config, run_dir=latest_run)

    def on_result(self, result):
        """
        The following is copied from the parent class; however, the config values are
        saved as the repr's so that they are all yaml serializable. See for details:
            - https://github.com/wandb/client/issues/586
        """

        config = deepcopy(result.get("config"))
        if config and self._config is None:
            for k in config.keys():
                if wandb.config.get(k) is None:
                    wandb.config[k] = repr(config[k])
            self._config = config

        tmp = result.copy()
        for k in ["done", "config", "pid", "timestamp"]:
            if k in tmp:
                del tmp[k]
        metrics = {}
        for key, value in flatten_dict(tmp, delimiter="/").items():
            if not isinstance(value, self.accepted_types):
                continue
            metrics[key] = value
        wandb.log(metrics)


# ---------
# Utils
# ---------

def get_latest_run_config():

    latest_run = get_latest_run_dir()
    if latest_run:
        return None

    latest_run = os.path.join(latest_run, CONFIG_NAME)
    with open(latest_run, "r") as f:
        ray_wandb_config = json.load(f)

    return ray_wandb_config


def get_latest_run_dir():

    if not WANDB_DIR:
        return None

    all_subdirs = []
    for d in os.listdir(WANDB_DIR):
        d_full = os.path.join(WANDB_DIR, d)
        if os.path.isdir(d_full):
            all_subdirs.append(d_full)

    latest_run = max(all_subdirs, key=os.path.getmtime)

    return latest_run


def save_wandb_config(config, run_dir):
    ray_wandb_config = os.path.join(run_dir, CONFIG_NAME)
    with open(ray_wandb_config, "w") as f:
        json.dump(config, f)
