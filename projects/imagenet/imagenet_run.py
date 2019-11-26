#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import datetime
import logging
import os
import time

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from experiments import CONFIGS
from torch.backends import cudnn

from nupic.research.frameworks.pytorch.model_utils import set_random_seed
from projects.imagenet.imagenet_experiment import ImagenetExperiment


cudnn.benchmark = True


def configure_logger(name, level, path):
    handler = logging.FileHandler(os.path.join(path, "experiment.log"))
    handler.setFormatter(
        logging.Formatter("%(asctime)s:%(name)s:%(levelname)s: %(message)s"))
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def main_worker(rank, world_size, config):
    """
    Distributed process worker
    :param rank: Process rank or id withing the group
    :param world_size: Total number of processes in the group
    :param config: Experiment configuration
    """
    distributed = config["distributed"]
    if distributed:
        # Update GPU configuration
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            config["device"] = rank
            config["workers"] = 2
            config["device_ids"] = [rank]

        # Initialize distributed process group for this rank/gpu/cpu
        dist.init_process_group(
            backend=config["backend"],
            init_method="tcp://127.0.0.1:54311",
            world_size=world_size,
            rank=rank,
        )
        # Explicitly setting seed to make sure that models start from same
        # random weights and biases across processes.
        set_random_seed(config.get("seed", 1), config.get("deterministic_mode", False))

    # Configure python logger for this process/rank
    name = config["name"]
    logdir = config["logdir"]
    logger = configure_logger("{}.{}".format(name, rank), config["loglevel"], logdir)
    logger.debug("Configuration %s", config)

    # Only show progress on the first process of the group
    config["progress_bar"] = config["progress_bar"] and rank == 0

    # Start training
    logger.info("Training started")
    results = []

    best_accuracy = 0.0
    stop_condition = config["stop"]
    epochs = config["epochs"]
    checkpoint_start = config["checkpoint_start"]
    exp = ImagenetExperiment()
    exp.setup(config)
    for epoch in range(epochs):
        start = time.time()
        lr = exp.get_lr()

        # Train on epoch
        exp.pre_epoch(epoch)
        exp.train(epoch)
        validation = exp.validate()
        exp.post_epoch(epoch)

        # Collect results
        accuracy = validation["mean_accuracy"]
        loss = validation["mean_loss"]
        duration = time.time() - start
        validation.update({
            "duration": duration,
            "lr": lr})
        results.append(validation)

        logger.info("Epoch %d: Accuracy=%f, Loss=%f, lr=%s, duration=%f",
                    epoch, accuracy, loss, lr, duration)

        # Save best model on the first process of the group
        if rank == 0 and accuracy > best_accuracy:
            # Wait a few epoch before saving checkpoints
            if epoch >= checkpoint_start:
                exp.save(logdir)
            best_accuracy = accuracy

        # Stop computing once accuracy reaches target accuracy
        if any(validation[k] >= stop_condition[k] for k in stop_condition.keys()):
            break

    exp.stop()
    logger.info("Training finished")

    # Save results on the first process of the group
    if rank == 0:
        df = pd.DataFrame(results)
        results_csv = os.path.join(logdir, "results.csv")
        df.to_csv(results_csv, index_label="epoch")

    if distributed:
        dist.destroy_process_group()


def main(args):
    # Get configuration
    name = args.name
    config = dict(CONFIGS[name])
    config.update(vars(args))

    # Prepare results path
    logdir = os.path.join(
        config["results"], name,
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    os.makedirs(logdir, exist_ok=True)
    config["logdir"] = logdir

    if args.distributed:
        # Select number of processes to match number of GPUs
        if torch.cuda.is_available():
            nprocs = torch.cuda.device_count()
        else:
            nprocs = torch.get_num_interop_threads() - 1
        mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, config,))
    else:
        main_worker(0, 1, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--experiment", dest="name",
                        help="Experiment to run", choices=CONFIGS.keys())
    parser.add_argument("-l", "--loglevel", default=logging.INFO,
                        help="Python logging level")
    parser.add_argument("-d", "--distributed", default=True,
                        help="Whether or not to use Pytorch Distributed")
    parser.add_argument("-b", "--backend", choices=["nccl", "gloo"],
                        help="Pytorch Distributed backend",
                        default="nccl" if torch.cuda.is_available() else "gloo")

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
    else:
        main(args)
