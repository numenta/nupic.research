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
import collections
import os
import pickle
import time
import uuid
from datetime import datetime

import torch.distributed as dist
import torch.multiprocessing as mp

from nupic.research.frameworks.pytorch.imagenet import ImagenetExperiment
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import get_free_port


# Disable HDF5 locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def worker(rank, world_size, dist_url, config, queue):
    """
    Main distributed training worker process
    """
    # Limit this process to a single GPU deriving GPU ID from worker rank
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # Update config with distributed parameters
    config["dist_url"] = dist_url
    config["rank"] = rank
    config["world_size"] = world_size
    config["distributed"] = True

    # Setup imagenet experiment for this process
    experiment_class = config.get("experiment_class", ImagenetExperiment)
    assert issubclass(experiment_class, ImagenetExperiment)
    exp = experiment_class()
    exp.setup_experiment(config)

    # Wait until all processes have finished setting up
    dist.barrier()

    # Check if restoring experiment from checkpoint
    checkpoint_file = config.get("checkpoint_file", None)
    if checkpoint_file is not None:
        with open(checkpoint_file, mode="rb") as f:
            state = pickle.load(f)
            exp.set_state(state)
            # Wait until all processes have finished loading the checkpoint
            dist.barrier()

    checkpoint_at_end = config.get("checkpoint_at_end", False)
    checkpoint_freq = config.get("checkpoint_freq", 0)

    # Run experiment
    while not exp.should_stop():
        result = exp.run_epoch()
        epoch = exp.get_current_epoch()
        should_checkpoint = False
        if rank == 0:
            if checkpoint_at_end and exp.should_stop():
                should_checkpoint = True
            elif checkpoint_freq > 0 and (epoch % checkpoint_freq) == 0:
                should_checkpoint = True

        checkpoint = exp.get_state() if should_checkpoint else None
        queue.put((epoch, result, checkpoint))

    exp.stop_experiment()

    # Signal this worker is done by returning None
    queue.put(None)


def run(config, logger=None, on_checkpoint=None):
    """
    Run ImagenetExperiment distributed training using torch.multiprocessing
    given it's configuration
    :param config: Experiment configuration
    :param logger: Optional result logger callback reporting the results on
                   each epoch. "function(result)"
    :param on_checkpoint: Optional checkpoint callback called whenever a
                          checkpoint is available. "function(epoch, checkpoint)"
    :return: List with all results.
    """
    if logger is not None:
        assert callable(logger)

    num_gpus = config.get("num_gpus", 0)
    num_cpus = config.get("num_cpus", 1)
    experiment_class = config.get("experiment_class", ImagenetExperiment)
    experiment_tag = config.pop("experiment_tag", None)
    experiment_id = uuid.uuid4().hex

    # Determine the number of distributed processes based on the number GPUs and CPUs
    if num_gpus > 0:
        world_size = num_gpus
    else:
        world_size = num_cpus

    # Launch worker processes using a SimpleQueue to collect the results
    dist_url = f"tcp://localhost:{get_free_port()}"
    queue = mp.SimpleQueue()
    ctx = mp.spawn(worker, args=(world_size, dist_url, config, queue),
                   nprocs=world_size, join=False)  # non-blocking

    # Collect worker results as they become available waiting until all worker
    # processes are done.
    worker_results = collections.defaultdict(list)
    results = []
    pending_workers = world_size
    t1 = t0 = time.time()
    while pending_workers > 0:
        item = queue.get()
        if item is None:
            # The worker process returns None when it is done
            pending_workers -= 1
        else:
            epoch, data, checkpoint = item
            worker_results[epoch].append(data)
            if on_checkpoint is not None and checkpoint is not None:
                on_checkpoint(epoch, checkpoint)

            # Aggregated worker results at the end of each epoch
            epoch_result = worker_results[epoch]
            if len(epoch_result) == world_size:
                t2 = time.time()
                epoch_result = experiment_class.aggregate_results(epoch_result)

                # Update ray.tune fields
                now = datetime.today()
                epoch_result.update(training_iteration=epoch,
                                    time_this_iter_s=t2 - t1,
                                    time_total_s=t2 - t0,
                                    date=now.strftime("%Y-%m-%d_%H-%M-%S"),
                                    timestamp=int(time.mktime(now.timetuple())),
                                    pid=os.getpid(),
                                    hostname=os.uname()[1],
                                    config=config,
                                    experiment_tag=experiment_tag,
                                    experiment_id=experiment_id,
                                    neg_mean_loss=-epoch_result["mean_loss"],
                                    timesteps_total=epoch_result.get("timestep", 0)
                                    )
                # Log epoch result
                results.append(epoch_result)
                if logger is not None:
                    logger(epoch_result)
                t1 = t2

    # Wait until all worker processes terminate
    ctx.join()

    return results
