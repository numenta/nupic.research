#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import argparse
import copy

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.imagenet import imagenet_tune
from nupic.research.frameworks.pytorch.imagenet.parser_utils import (
    DEFAULT_PARSERS,
    process_args,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=DEFAULT_PARSERS,
    )
    parser.add_argument("-e", "--experiment", dest="name", default="default_base",
                        help="Experiment to run", choices=CONFIGS.keys())

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    # Get configuration values
    config = copy.deepcopy(CONFIGS[args.name])

    # Merge configuration with command line arguments
    config.update(vars(args))

    # Process args and modify config appropriately.
    config = process_args(args, config)

    if config is None:
        pass
    elif "single_instance" in args:
        imagenet_tune.run_single_instance(config)
    else:
        imagenet_tune.run(config)
