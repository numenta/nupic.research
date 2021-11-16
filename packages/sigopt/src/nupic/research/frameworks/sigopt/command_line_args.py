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

from nupic.research.frameworks.sigopt import SigOptExperiment

__all__ = [
    "get_parser",
    "process_args",
]


def get_parser():
    """
    Returns command line `argparse.ArgumentParser` with sigopt command line
    arguments
    """

    sigopt_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    sigopt_parser.add_argument("-T", "--create_sigopt", action="store_true",
                               help="Create a new sigopt experiment using the config")

    return sigopt_parser


def process_args(args, config):
    """
    Processes parsed arguments to modify config appropriately.
    :return: modified config or None to exit without running
    """
    if "create_sigopt" in args:
        s = SigOptExperiment()
        s.create_experiment(config["sigopt_config"])
        print("Created experiment: https://app.sigopt.com/experiment/",
              s.experiment_id)
        return None
    return config
