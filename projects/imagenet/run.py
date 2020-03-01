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
import socket

import torch

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.imagenet import imagenet_tune
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptImagenetExperiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument("-e", "--experiment", dest="name", default="default",
                        help="Experiment to run", choices=CONFIGS.keys())
    parser.add_argument("-g", "--num-gpus", type=int,
                        default=torch.cuda.device_count(),
                        help="number of GPUs to use")
    parser.add_argument("-n", "--num-cpus", type=int,
                        default=torch.get_num_interop_threads(),
                        help="number of CPUs to use when GPU is not available."),
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Resume training from last known checkpoint")
    parser.add_argument("-j", "--workers", type=int, default=6,
                        help="Number of dataloaders workers")
    parser.add_argument("-b", "--backend", choices=["nccl", "gloo"],
                        help="Pytorch Distributed backend", default="nccl")
    parser.add_argument("-d", "--dist-port", type=int, default=54321,
                        help="tcp port to use for distributed pytorch training")
    parser.add_argument("-s", "--with-server", action="store_true",
                        help="Start Ray Tune API server")
    parser.add_argument("-p", "--progress", action="store_true",
                        help="Show progress during training")
    parser.add_argument("-l", "--log-level",
                        choices=["critical", "error", "warning", "info", "debug"],
                        help="Python Logging level")
    parser.add_argument("-f", "--log-format",
                        help="Python Logging Format")
    parser.add_argument("-x", "--max-failures", type=int, default=1,
                        help="How many times to try to recover before stopping")
    parser.add_argument("-c", "--checkpoint-freq", type=int,
                        help="How often to checkpoint (epochs)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.autograd.profiler.profile during training")
    parser.add_argument("-t", "--create_sigopt", action="store_true",
                        help="Create a new sigopt experiment using the config")
    parser.add_argument(
        "-a", "--redis-address",
        default="{}:6379".format(socket.gethostbyname(socket.gethostname())),
        help="redis address of an existing Ray server")

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    # Get configuration values
    config = copy.deepcopy(CONFIGS[args.name])

    # Merge configuration with command line arguments
    config.update(vars(args))

    if "create_sigopt" in args:
        s = SigOptImagenetExperiment()
        s.create_experiment(config["sigopt_config"])
        print(
            "Created experiment: https://app.sigopt.com/experiment/"
            + str(s.experiment_id))
    else:
        imagenet_tune.run(config)
