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

import ray  # noqa: F401
from ray import tune

from experiments import CONFIGS
from mixins import GSCNoiseTest
from nupic.research.frameworks.ray.run_with_raytune import run
from nupic.research.frameworks.vernon.parser_utils import (
    get_default_parsers,
    insert_experiment_mixin,
    process_args,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=get_default_parsers(),
    )
    parser.add_argument("-e", "--experiment", dest="name", default="default_base",
                        help="Experiment to run", choices=CONFIGS.keys())
    parser.add_argument("--evaluate-noise", action="store_true",
                        help="Whether or not to run noise tests")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to run")
    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    # Get configuration values
    config = copy.deepcopy(CONFIGS[args.name])

    # Merge configuration with command line arguments
    config.update(vars(args))

    # Add noise tests
    if args.evaluate_noise:
        insert_experiment_mixin(config=config, mixin=GSCNoiseTest)

    # Replace static seed if using multiple samples
    if args.num_samples > 1:
        seed = config.get("seed", 0)
        if isinstance(seed, int):
            config.update(seed=tune.randint(args.num_samples * 100))

    # Process args and modify config appropriately.
    config = process_args(args, config)
    if config is None:
        pass
    else:
        run(config)
