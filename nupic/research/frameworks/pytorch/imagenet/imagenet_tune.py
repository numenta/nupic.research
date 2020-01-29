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
import copy
import os
from pprint import pprint

import ray
from ray.tune import Trainable, tune
from ray.tune.resources import Resources

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.imagenet import ImagenetExperiment

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


class ImagenetTrainable(Trainable):
    """
    Trainable class used to train resnet50 on Imagenet dataset using ray
    """

    @classmethod
    def default_resource_request(cls, config):
        """
        Configure the cluster resources used by this experiment
        """
        num_gpus = config.get("num_gpus", 0)
        num_cpus = config.get("num_cpus", 1)

        if num_gpus > 0:
            # Assign extra CPUs for dataloaders
            workers = config.get("workers", 0)
            num_cpus = workers * num_gpus

        resource = Resources(cpu=0, gpu=0, extra_cpu=num_cpus, extra_gpu=num_gpus)
        return resource

    def _setup(self, config):
        num_gpus = config.get("num_gpus", 0)
        num_cpus = config.get("num_cpus", 1)

        # Determine the number of distributed processes based on the number
        # GPUs and CPUs
        if num_gpus > 0:
            world_size = num_gpus
            # Assign one GPU per remote process
            num_gpus = 1
            # Assign extra CPUs for dataloaders
            num_cpus = config.get("workers", 0)
        else:
            world_size = num_cpus
            # Assign one CPU per remote process
            num_cpus = 1

        # Create one ray remote process for each experiment in the process group
        experiment = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(
            ImagenetExperiment
        )
        self.procs = [experiment.remote() for _ in range(world_size)]

        # Use first process as head of the group
        ip = ray.get(self.procs[0].get_node_ip.remote())
        port = config.get("dist_port", 54321)
        dist_url = "tcp://{}:{}".format(ip, port)

        # Configure each process in the group
        status = []
        for i, w in enumerate(self.procs):
            worker_config = copy.deepcopy(config)
            worker_config["distributed"] = True
            worker_config["dist_url"] = dist_url
            worker_config["world_size"] = world_size
            worker_config["rank"] = i
            status.append(w.setup_experiment.remote(worker_config))

        # Wait for remote functions to complete
        ray.get(status)

    def _train(self):
        status = []
        for w in self.procs:
            status.append(w.run_epoch.remote(self.iteration))

        # Wait for remote functions to complete
        results = ray.get(status)

        # Return the results from the first remote function
        return copy.deepcopy(results[0])

    def _save(self, _):
        # All models are synchronized. Just save the state of first model
        return ray.get(self.procs[0].get_state.remote())

    def _restore(self, state):
        # Restore the state to every process
        state_id = ray.put(state)
        ray.get([w.set_state.remote(state_id) for w in self.procs])

    def _stop(self):
        for w in self.procs:
            w.stop_experiment.remote()
            w.__ray_terminate__.remote()


def run(config):
    # Connect to ray
    address = os.environ.get("REDIS_ADDRESS", config.get("redis_address"))
    ray.init(address=address)

    # Build kwargs for `tune.run` function using merged config and command line dict
    kwargs_names = tune.run.__code__.co_varnames[:tune.run.__code__.co_argcount]
    kwargs = dict(zip(kwargs_names, [ImagenetTrainable, *tune.run.__defaults__]))

    # Update`tune.run` kwargs with config
    kwargs.update(config)
    kwargs["config"] = config

    # Update tune stop criteria with config epochs
    stop = kwargs.get("stop", {}) or dict()
    epochs = config.get("epochs", 1)
    stop.update(training_iteration=epochs)
    kwargs["stop"] = stop

    # Make sure to only select`tune.run` function arguments
    kwargs = dict(filter(lambda x: x[0] in kwargs_names, kwargs.items()))

    # Queue trials until the cluster scales up
    kwargs.update(queue_trials=True)

    pprint(kwargs)
    tune.run(**kwargs)
    ray.shutdown()
