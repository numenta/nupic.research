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
from datetime import datetime
from functools import partial

import torch
import torch.multiprocessing as multiprocessing

from nupic.research.frameworks.pytorch.imagenet import imagenet_run
from nupic.research.frameworks.pytorch.imagenet import (
    mixins, ImagenetExperiment
)
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptImagenetExperiment


multiprocessing.set_start_method("spawn", force=True)


def insert_experiment_mixin(config, mixin):
    experiment_class = config["experiment_class"]


    class Cls(mixin, experiment_class):
        pass


    Cls.__name__ = f"{mixin.__name__}{experiment_class.__name__}"
    config["experiment_class"] = Cls


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


def run_trial(trial_config):
    """
    Run a single trial configuration
    """
    # Configure ray.tune loggers
    from ray.tune.logger import UnifiedLogger
    logger = UnifiedLogger(config=trial_config,
                           logdir=trial_config["logdir"],
                           loggers=trial_config.get("loggers", None))

    result = imagenet_run.run(config=trial_config,
                              logger=logger.on_result,
                              on_checkpoint=partial(save_checkpoint, trial_config))

    logger.flush()
    logger.close()
    return result


if __name__ == "__main__":
    # The spawned 'imagenet_run.run' process does not import 'ray' however some
    # configurations in the experiment package import 'ray' and 'ray.tune'. When
    # 'ray' is imported after 'pickle' it throws an exception. We avoid this
    # exception by loading the configurations in "main" local context instead of
    # the global context
    from experiments import CONFIGS

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        description=__doc__
    )
    parser.add_argument("-e", "--experiment", dest="name",
                        help="Experiment to run", choices=CONFIGS.keys())
    parser.add_argument("-g", "--num-gpus", type=int,
                        default=torch.cuda.device_count(),
                        help="number of GPUs to use")
    parser.add_argument("-n", "--num-cpus", type=int,
                        default=torch.get_num_interop_threads(),
                        help="number of CPUs to use when GPU is not available."),
    parser.add_argument("-c", "--checkpoint-file", dest="restore_checkpoint_file",
                        help="Resume experiment from specific checkpoint file")
    parser.add_argument("-j", "--workers", type=int, default=6,
                        help="Number of dataloaders workers")
    parser.add_argument("-b", "--backend", choices=["nccl", "gloo"],
                        help="Pytorch Distributed backend", default="nccl")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="Show progress during training")
    parser.add_argument("-l", "--log-level",
                        choices=["critical", "error", "warning", "info", "debug"],
                        help="Python Logging level")
    parser.add_argument("-f", "--log-format",
                        help="Python Logging Format")
    parser.add_argument("-x", "--max-failures", type=int,
                        help="How many times to try to recover before stopping")
    parser.add_argument("--checkpoint-freq", type=int,
                        help="How often to checkpoint (epochs)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.autograd.profiler.profile during training")
    parser.add_argument("--profile-autograd", action="store_true",
                        help="Enable torch.autograd.profiler.profile during training")
    parser.add_argument("-t", "--create_sigopt", action="store_true",
                        help="Create a new sigopt experiment using the config")

    args = parser.parse_args()
    if "name" not in args:
        parser.print_help()
        exit(1)

    # Get configuration values
    config = copy.deepcopy(CONFIGS[args.name])

    # Merge configuration with command line arguments
    config.update(vars(args))

    if "profile" in args and args.profile:
        insert_experiment_mixin(config, mixins.Profile)

    if "profile_autograd" in args and args.profile_autograd:
        insert_experiment_mixin(config, mixins.ProfileAutograd)

    if "create_sigopt" in args:
        s = SigOptImagenetExperiment()
        s.create_experiment(config["sigopt_config"])
        print(
            "Created experiment: https://app.sigopt.com/experiment/"
            + str(s.experiment_id))

    results = []
    for trial in create_trials(config):
        results.append(run_trial(trial))
