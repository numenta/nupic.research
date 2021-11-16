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
from importlib import metadata

import torch

from nupic.research.frameworks.vernon import mixins

__all__ = [
    "get_default_parsers",
    "insert_experiment_mixin",
    "process_args",
]


def _get_frameworks_command_line_args_():
    """
    Get installed frameworks extentions to command line arguments.
    Each framework willing to extend the command line args must implement a
    module with the following functions:
        def get_parser() -> argparse.ArgumentParser:
        def process_args(args: argpase.Namespace, config: dict) -> dict:
    And advertise the extension module using an entry point named "command_line_args"
    under the "nupic.research.frameworks" entry point group. For example::

    setups.cfg:
        [options.entry_points]
        nupic.research.frameworks =
            command_line_args = nupic.research.frameworks.ray.command_line_args

    see:: https://setuptools.pypa.io/en/latest/userguide/entry_point.html

    """
    # Load currently installed nupic.research.frameworks with plugins
    return [
        ep.load() for ep in metadata.entry_points()["nupic.research.frameworks"]
        if ep.name == "command_line_args"
    ]


def get_default_parsers():
    """
    Returns command line `argparse.ArgumentParser` with vernon default options
    as well as the command line options for any installed nupic.research.framework
    """
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    main_parser.add_argument("-g", "--num-gpus", type=int,
                             default=torch.cuda.device_count(),
                             help="number of GPUs to use")
    main_parser.add_argument("-n", "--num-cpus", type=int,
                             default=torch.get_num_interop_threads(),
                             help="number of CPUs to use when GPU is not available."),
    main_parser.add_argument("-r", "--restore", action="store_true",
                             help="Restore training from last known checkpoint")
    main_parser.add_argument("-c", "--checkpoint-file", dest="restore_checkpoint_file",
                             help="Resume experiment from specific checkpoint file")
    main_parser.add_argument("-d", "--copy-checkpoint-to-dir",
                             dest="copy_checkpoint_dir",
                             help="Copy final saved checkpoint to specified directory.")
    main_parser.add_argument("-j", "--workers", type=int, default=4,
                             help="Number of dataloaders workers")
    main_parser.add_argument("-b", "--backend", choices=["nccl", "gloo"],
                             help="Pytorch Distributed backend", default="nccl")
    main_parser.add_argument("-p", "--progress", action="store_true",
                             help="Show progress during training")
    main_parser.add_argument("-l", "--log-level",
                             choices=["critical", "error", "warning", "info", "debug"],
                             help="Python Logging level")
    main_parser.add_argument("-f", "--log-format",
                             help="Python Logging Format")
    main_parser.add_argument("-x", "--max-failures", type=int,
                             help="How many times to try to recover before stopping")
    main_parser.add_argument("--checkpoint-freq", type=int,
                             help="How often to checkpoint (epochs)")
    main_parser.add_argument("--profile", action="store_true",
                             help="Enable cProfile tracing")
    main_parser.add_argument("--profile-autograd", action="store_true",
                             help="Enable torch.autograd.profiler.profile during"
                                  " training")

    parsers = [main_parser]
    installed_frameworks = _get_frameworks_command_line_args_()
    for framework in installed_frameworks:
        parsers.append(framework.get_parser())
    return parsers


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

    installed_frameworks = _get_frameworks_command_line_args_()
    for entry_point in installed_frameworks:
        config = entry_point.process_args(args, config)
        if config is None:
            # Return None to exit. Used by sigopt when the experiments is created
            return None

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
