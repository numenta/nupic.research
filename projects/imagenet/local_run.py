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
"""
Use this script to run Imagenet Experiments on a single node using Distributed
Data Parallel (DDP) without ray
"""
import argparse
import copy
import os
import tempfile
import time
import uuid
from datetime import datetime
from functools import partial

# FIXME: When 'ray' is imported after 'pickle' it throws an exception.
import ray  # noqa: F401, I001
import torch.multiprocessing as multiprocessing

from experiments import CONFIGS
from nupic.research.frameworks import vernon
from nupic.research.frameworks.vernon import ImagenetExperiment
from nupic.research.frameworks.vernon.parser_utils import MAIN_PARSER, process_args

multiprocessing.set_start_method("spawn", force=True)


def create_trials(config):
    """
    Create trial configuration for each trial variant evaluating 'ray.tune'
    functions (grid_search, sample_from, ...) into its final values and
    creating the local and log dir for each trial

    :param config: Ray tune configuration with 'ray.tune' functions
    :return: list of dict for each trial configuration variant
    """
    from nupic.research.support.ray_utils import generate_trial_variants
    trials = generate_trial_variants(config)
    timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    for variant in trials:
        variant["experiment_id"] = uuid.uuid4().hex

        # Create local dir
        local_dir = os.path.join(
            os.path.expanduser(variant["local_dir"]), variant["name"])
        variant["local_dir"] = local_dir
        os.makedirs(local_dir, exist_ok=True)

        # Create logdir
        experiment_class = variant.get("experiment_class", ImagenetExperiment)
        experiment_tag = variant["experiment_tag"]
        log_prefix = f"{experiment_class.__name__}_{experiment_tag}_{timestamp}"
        logdir = tempfile.mkdtemp(dir=local_dir, prefix=log_prefix)
        variant["logdir"] = logdir

    return trials


def save_checkpoint(config, epoch, checkpoint):
    """
    Callback responsible to save experiment checkpoint.
    It will store the checkpoint in the same location as ray.tune
    """
    import pickle
    logdir = config["logdir"]
    checkpoint_dir = os.path.join(logdir, f"checkpoint_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint")
    with open(checkpoint_file, mode="wb") as f:
        pickle.dump(checkpoint, f)


def log_results(logger, config, results):
    # Update ray.tune fields
    timestamp = results["timestamp"]
    results["config"] = config
    results["experiment_id"] = config["experiment_id"]
    results["experiment_tag"] = config["experiment_tag"]
    results["training_iteration"] = results.pop("epoch")
    results["neg_mean_loss"] = results["mean_loss"]
    results["timesteps_total"] = results.get("timestep", 0)
    results["timestamp"] = int(timestamp)
    results["date"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
    results["time_this_iter_s"] = timestamp - logger.last_timestamp
    results["time_total_s"] = timestamp - logger.start_timestamp
    logger.on_result(results)
    logger.last_timestamp = timestamp


def run_trial(config):
    """
    Run a single trial configuration
    """
    # Configure ray.tune loggers
    from ray.tune.logger import UnifiedLogger
    logger = UnifiedLogger(config=config,
                           logdir=config["logdir"],
                           loggers=config.get("loggers", None))
    logger.last_timestamp = logger.start_timestamp = time.time()

    result = vernon.run(config=config,
                        logger=partial(log_results, logger, config),
                        on_checkpoint=partial(save_checkpoint, config))

    logger.flush()
    logger.close()
    return result


def main(args):
    # Get configuration values
    config = copy.deepcopy(CONFIGS[args.name])

    # Merge configuration with command line arguments
    config.update(vars(args))

    config = process_args(args, config)
    if config is None:
        # This may return when a sigopt experiment is created.
        print("Nothing to run (config=None).")
        return

    results = []
    for trial in create_trials(config):
        results.append(run_trial(trial))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[MAIN_PARSER],
        argument_default=argparse.SUPPRESS,
        description=__doc__
    )
    parser.add_argument("-e", "--experiment", dest="name",
                        help="Experiment to run", choices=CONFIGS.keys())

    args = parser.parse_args()
    if "name" not in args:
        parser.print_help()
        exit(1)
    main(args)
