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

import argparse
import socket

import torch
from wandb import util

from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers_2 import (
    DEFAULT_LOGGERS,
)
from nupic.research.frameworks.sigopt import SigOptExperiment
from nupic.research.frameworks.vernon import mixins
from nupic.research.frameworks.wandb import ray_wandb

__all__ = [
    "DEFAULT_PARSERS",
    "MAIN_PARSER",
    "RAY_PARSER",
    "SIGOPT_PARSER",
    "WANDB_PARSER",
    "process_args",
    "insert_experiment_mixin",
]


MAIN_PARSER = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    argument_default=argparse.SUPPRESS,
    add_help=False,
)
MAIN_PARSER.add_argument("-g", "--num-gpus", type=int,
                         default=torch.cuda.device_count(),
                         help="number of GPUs to use")
MAIN_PARSER.add_argument("-n", "--num-cpus", type=int,
                         default=torch.get_num_interop_threads(),
                         help="number of CPUs to use when GPU is not available."),
MAIN_PARSER.add_argument("-r", "--restore", action="store_true",
                         help="Restore training from last known checkpoint")
MAIN_PARSER.add_argument("-c", "--checkpoint-file", dest="restore_checkpoint_file",
                         help="Resume experiment from specific checkpoint file")
MAIN_PARSER.add_argument("-d", "--copy-checkpoint-to-dir", dest="copy_checkpoint_dir",
                         help="Copy final saved checkpoint to specified directory.")
MAIN_PARSER.add_argument("-j", "--workers", type=int, default=4,
                         help="Number of dataloaders workers")
MAIN_PARSER.add_argument("-b", "--backend", choices=["nccl", "gloo"],
                         help="Pytorch Distributed backend", default="nccl")
MAIN_PARSER.add_argument("-p", "--progress", action="store_true",
                         help="Show progress during training")
MAIN_PARSER.add_argument("-l", "--log-level",
                         choices=["critical", "error", "warning", "info", "debug"],
                         help="Python Logging level")
MAIN_PARSER.add_argument("-f", "--log-format",
                         help="Python Logging Format")
MAIN_PARSER.add_argument("-x", "--max-failures", type=int,
                         help="How many times to try to recover before stopping")
MAIN_PARSER.add_argument("--checkpoint-freq", type=int,
                         help="How often to checkpoint (epochs)")
MAIN_PARSER.add_argument("--profile", action="store_true",
                         help="Enable cProfile tracing")
MAIN_PARSER.add_argument("--profile-autograd", action="store_true",
                         help="Enable torch.autograd.profiler.profile during training")


RAY_PARSER = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    argument_default=argparse.SUPPRESS,
    add_help=False,
)
RAY_PARSER.add_argument("-s", "--with-server", action="store_true",
                        help="Start Ray Tune API server")
RAY_PARSER.add_argument("--single_instance", action="store_true",
                        help="Uses single instance run method")
RAY_PARSER.add_argument("--local-mode", action="store_true",
                        help="Start ray in local mode. Useful for debugging")
RAY_PARSER.add_argument("-a", "--redis-address",
                        help="redis address of an existing Ray server",
                        default="{}:6379".format(
                            socket.gethostbyname(socket.gethostname())
                        ))


SIGOPT_PARSER = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    argument_default=argparse.SUPPRESS,
    add_help=False,
)
SIGOPT_PARSER.add_argument("-t", "--create_sigopt", action="store_true",
                           help="Create a new sigopt experiment using the config")


WANDB_PARSER = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    argument_default=argparse.SUPPRESS,
    add_help=False,
)
WANDB_PARSER.add_argument("--wandb", action="store_true",
                          help="Enable logging through wandb.")
WANDB_PARSER.add_argument("--wandb_resume", action="store_true",
                          help="Resume logging through wandb.")


DEFAULT_PARSERS = [
    MAIN_PARSER,
    RAY_PARSER,
    SIGOPT_PARSER,
    WANDB_PARSER,
]


def process_args(args, config):
    """
    Processes parsed arguments to modify config appropriately.

    This returns None when `create_sigopt` is included in the args
    signifying there is nothing to run.

    :return: modified config or None
    """

    if "profile" in args and args.profile:
        insert_experiment_mixin(config, mixins.Profile)

    if "profile_autograd" in args and args.profile_autograd:
        insert_experiment_mixin(config, mixins.ProfileAutograd)

    if "copy_checkpoint_dir" in args:
        config["copy_checkpoint_dir"] = args.copy_checkpoint_dir
        insert_experiment_mixin(
            config, mixins.SaveFinalCheckpoint, prepend_name=False
        )

    if "wandb" in args and args.wandb:

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
        insert_experiment_mixin(config, ray_wandb.WorkerLogger, prepend_name=False)

    if "create_sigopt" in args:
        s = SigOptExperiment()
        s.create_experiment(config["sigopt_config"])
        print("Created experiment: https://app.sigopt.com/experiment/",
              s.experiment_id)
        return

    return config


def insert_experiment_mixin(config, mixin, prepend_name=True):
    experiment_class = config["experiment_class"]

    class Cls(mixin, experiment_class):
        pass

    if prepend_name:
        Cls.__name__ = f"{mixin.__name__}{experiment_class.__name__}"
    else:
        Cls.__name__ = experiment_class.__name__

    config["experiment_class"] = Cls
