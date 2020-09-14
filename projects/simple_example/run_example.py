#  Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc.  No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.

import argparse
import copy

from simple_example import CONFIGS
from nupic.research.frameworks.vernon.run_experiment import run_with_raytune
from nupic.research.frameworks.vernon.parser_utils import (
    DEFAULT_PARSERS,
    process_args,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=DEFAULT_PARSERS,
    )
    parser.add_argument("-e", "--experiment", nargs="+", dest="names",
                        default="default", help="Experiment to run",
                        choices=CONFIGS.keys())

    args = parser.parse_args()
    if args.names is None:
        parser.print_help()
        exit(1)

    # Get configuration values
    for name in args.names:
        config = copy.deepcopy(CONFIGS[name])

        # Merge configuration with command line arguments
        config.update(vars(args))
        config["name"] = name
        del config["names"]

        # Process args and modify config appropriately.
        config = process_args(args, config)

        # Run the config.
        if config is None:
            pass
        elif "single_instance" in args:
            run_with_raytune.run_single_instance(config)
        else:
            run_with_raytune.run(config)
