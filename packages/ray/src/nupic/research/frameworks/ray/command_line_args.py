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
import socket

__all__ = [
    "get_parser",
    "process_args",
]


def get_parser():
    """
    Returns command line `argparse.ArgumentParser` with ray and ray tune command
    line arguments
    """
    ray_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )
    ray_parser.add_argument("-s", "--with-server", action="store_true",
                            help="Start Ray Tune API server")
    ray_parser.add_argument("--single_instance", action="store_true",
                            help="Uses single instance run method")
    ray_parser.add_argument("--local-mode", action="store_true",
                            help="Start ray in local mode. Useful for debugging")
    ray_parser.add_argument("-a", "--redis-address",
                            help="redis address of an existing Ray server",
                            default="{}:6379".format(
                                socket.gethostbyname(socket.gethostname())
                            ))
    return ray_parser


def process_args(args, config):
    """
    Processes parsed arguments to modify config appropriately.
    :return: modified config or None to exit without running
    """
    return config
