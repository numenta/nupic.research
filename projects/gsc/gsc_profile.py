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
import sys
from os.path import expanduser
from pprint import pprint
from time import time

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader

from experiments import CONFIGS  # noqa
from nupic.research.frameworks.pytorch.datasets import preprocessed_gsc
from nupic.research.frameworks.pytorch.restore_utils import load_state_from_checkpoint

PATH_TO_CONFIGS = expanduser("~/nta/nupic.research/projects/gsc")
sys.path.insert(0, PATH_TO_CONFIGS)

# Default settings.
DATA_ROOT = expanduser("~/nta/data/gsc_preprocessed")


def evaluate_model(model, dataloader):
    """
    Simply run a forward pass through all of the data; don't compute the
    loss or accuracy. All inference is done on GPU.
    """
    samples_processed = 0
    for image, _ in dataloader:
        samples_processed += len(image)
        image = image.to("cuda")
        model(image)
    return samples_processed


def profile(model, num_workers, num_runs, batch_size):
    # Load the train dataset and accompanying dataloader.
    train_dataset = preprocessed_gsc(root=DATA_ROOT, train=True, download=False)
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,  # TODO: Will this affect timing?
    )

    print(f"Profiling inference with batch size: {batch_size}")
    run_times = []
    samples_processed = []
    for _ in range(num_runs + 1):
        t0 = time()
        s = evaluate_model(model, dataloader)
        run_times.append(time() - t0)
        samples_processed.append(s)

    # Compute and print statistics. We always ignore the first run to account for
    # startup time
    mu_time = np.mean(run_times[1:])
    std_time = np.std(run_times[1:])
    throughput = samples_processed[0] / mu_time
    print(f"Run times: {mu_time:0.4f} ± {std_time:0.4f} "
          f"averaged over {args.num_runs - 1} trials.")
    print("All run times:", run_times)
    print("Samples processed:", samples_processed)
    print(f"Throughput words/sec: {throughput:0.2f} "
          f"with batch size: {batch_size}")

    return throughput, mu_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
        add_help=False,
    )

    parser.add_argument("-e", "--experiment", dest="name",
                        help="Name of experiment config.", choices=CONFIGS.keys())
    parser.add_argument("-c", "--checkpoint-file", dest="checkpoint_file",
                        help="Path to checkpoint.")
    parser.add_argument("-n", "--num_runs", type=int, default=10,
                        help="Number of runs to average over.")
    parser.add_argument("-w", "--num_workers", type=int, default=0,
                        help="Number of workers for the dataloader.")
    parser.add_argument("-f", "--table_format", default="grid",
                        help="Table format: grid or latex.")
    parser.add_argument("-g", "--gpu", default="????",
                        help="Name of GPU")

    args = parser.parse_args()
    if args.checkpoint_file is None or args.name is None:
        parser.print_help()
        exit(1)

    print("Running inference with args:")
    pprint(vars(args))
    print()

    # Create and load model.
    config = CONFIGS[args.name]
    exp = config["experiment_class"]()
    model = exp.create_model(config, device="cuda")
    load_state_from_checkpoint(
        model,
        checkpoint_path=args.checkpoint_file,
        device="cuda"
    )
    print("Loaded model:\n", model)
    print()

    results = [
        [
            "GPU",
            "Batch size",
            "Num workers",
            "Num runs",
            "Throughput",
            "Mean run time",
        ]
    ]

    for batch_size in [4, 16, 64, 256, 1024, 8192]:
        throughput, run_time = profile(model, args.num_workers, args.num_runs,
                                       batch_size)
        results.append([args.gpu, batch_size, args.num_workers, args.num_runs,
                        throughput, run_time])
    print(tabulate(results, headers="firstrow", tablefmt=args.table_format,
                   stralign="left", numalign="center"))
