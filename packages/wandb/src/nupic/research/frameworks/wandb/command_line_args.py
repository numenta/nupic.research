# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
"""
This module is used as a nupic.research.frameworks plugin entrypoint to vernon
command line parser interface.
Each nupic.research.framework willing to add command line arguments to vernon
framework must implement two functions::
    - get_parser : Returns preconfigured `argparse.ArgumentParser` class to be
                   added to the main `argparse.ArgumentParser`.
    - process_args : Processes parsed arguments to modify config appropriately.

See nupic.research.frameworks.vernon.parset_utils for more details
"""
import argparse

from wandb import util

from nupic.research.frameworks.ray.ray_custom_loggers import DEFAULT_LOGGERS
from nupic.research.frameworks.vernon.parser_utils import insert_experiment_mixin
from nupic.research.frameworks.wandb import ray_wandb

__all__ = [
    "get_parser",
    "process_args",
]


def get_parser():
    """
    Returns command line `argparse.ArgumentParser` with wandb command line
    arguments
    """
    wandb_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    wandb_parser.add_argument("--wandb", action="store_true",
                              help="Enable logging through wandb.")
    wandb_parser.add_argument("--wandb_resume", action="store_true",
                              help="Resume logging through wandb.")
    return wandb_parser


def process_args(args, config):
    """
    Processes parsed arguments to modify config appropriately.
    :return: modified config or None to exit without running
    """
    if "wandb" in args:
        # Add ray-wandb logger to loggers.
        config.setdefault("loggers", [])
        config["loggers"].extend(list(DEFAULT_LOGGERS) + [ray_wandb.WandbLogger])

        # One may specify `wandb_args` or `env_config["wandb"]`
        name = config.get("name", "unknown_name")
        wandb_args = config.get("wandb_args", {})
        wandb_args.setdefault("name", name)
        config.setdefault("env_config", {})
        config["env_config"].setdefault("wandb", wandb_args)

        # Either restore from a run-id generate a new one.
        resume = wandb_args.get("resume", False)
        if ("wandb_resume" in args and args.wandb_resume) or resume:
            wandb_args.setdefault("resume", True)
            ray_wandb.enable_run_resume(wandb_args)
        else:
            wandb_id = util.generate_id()
            wandb_args["id"] = wandb_id

        # Enable logging on workers.
        insert_experiment_mixin(config, ray_wandb.PrepPlotForWandb, prepend_name=False)

    return config
