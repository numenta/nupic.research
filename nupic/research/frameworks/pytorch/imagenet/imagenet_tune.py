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
import logging
import os
import traceback
from pprint import pprint

import ray
import ray.resource_spec
from ray import ray_constants
from ray.exceptions import RayTimeoutError
from ray.tune import Trainable, tune
from ray.tune.ray_trial_executor import RESOURCE_REFRESH_PERIOD, RayTrialExecutor
from ray.tune.resources import Resources

from nupic.research.frameworks.pytorch.imagenet import ImagenetExperiment
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptImagenetExperiment

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
logger = logging.getLogger(__name__)
TRIAL_START_ATTEMPTS = 3
NODE_CREATION_TIMEOUT = 10 * 60


def _get_resources_per_node():
    """
    Maps node id to available resources on that node
    :return: dict with available :class:`Resources` for each node
    """
    def _is_node_key(k):
        return k.startswith(ray.resource_spec.NODE_ID_PREFIX)

    # Only consider active/alive nodes
    nodes = filter(lambda node: node["Alive"], ray.nodes())
    resources = map(lambda node: node["Resources"], nodes)

    # Group resources by node
    resources_by_node = {}
    for item in resources:
        node_id = next(filter(_is_node_key, item.keys()))

        item = item.copy()
        num_cpus = item.pop("CPU", 0)
        num_gpus = item.pop("GPU", 0)
        memory = ray_constants.from_memory_units(item.pop("memory", 0))
        object_store_memory = ray_constants.from_memory_units(
            item.pop("object_store_memory", 0))
        custom_resources = item

        resources_by_node[node_id] = Resources(
            int(num_cpus),
            int(num_gpus),
            memory=int(memory),
            object_store_memory=int(object_store_memory),
            custom_resources=custom_resources)

    return resources_by_node


class AffinityExecutor(RayTrialExecutor):
    """
    Ray Trial Executor used to set node affinity to node trials.
    This trial executor will update the trial configuration with one extra
    parameter (__ray_node_affinity__) that can be used to launch extra workers
    on the same node.
    """

    def __init__(self, queue_trials=False, reuse_actors=False, ray_auto_init=False,
                 refresh_period=RESOURCE_REFRESH_PERIOD):
        self._resources_by_node = {}
        super().__init__(queue_trials, reuse_actors, ray_auto_init, refresh_period)

    def _commit_resources(self, resources):
        resources = copy.deepcopy(resources)
        # Make sure "custom_resources" keys match "extra_custom_resources" keys
        # because `RayTrialExecutor._commit_resources` will only commit
        # resources from "extra_custom_resources" if the same resource key
        # is present in the "custom_resources" dict
        for k in resources.extra_custom_resources:
            if k not in resources.custom_resources:
                resources.custom_resources[k] = 0.0

        super()._commit_resources(resources)

    def _update_avail_resources(self, num_retries=5):
        super()._update_avail_resources(num_retries)
        self._resources_by_node = _get_resources_per_node()

    def _get_node_resources(self, resources):
        self._update_avail_resources()

        # Check for required GPUs first
        required = resources.gpu_total()
        if required > 0:
            resource_attr = "gpu"
        else:
            # No GPU, just use CPU
            resource_attr = "cpu"
            required = resources.cpu_total()

        # Compute nodes required to fulfill trial resources request
        custom_resources = {}
        for node_id, node_resource in self._resources_by_node.items():
            # Compute resource remaining on each node
            node_capacity = node_resource.get_res_total(node_id)
            committed_capacity = self._committed_resources.get_res_total(node_id)
            remaining_capacity = node_capacity - committed_capacity
            node_procs = getattr(node_resource, resource_attr, 0)
            available = node_procs * remaining_capacity

            if available == 0:
                continue

            if required <= available:
                custom_resources[node_id] = required / node_procs
                required = 0
                break
            else:
                custom_resources[node_id] = remaining_capacity
                required -= available

        if required > 0:
            # Not enough nodes
            return None

        return custom_resources

    def start_trial(self, trial, checkpoint=None):
        # Reserve node before starting trial
        resources = trial.resources

        # Use no-op remote with same resource requirements as trial to trick
        # ray into scale up the cluster before we allocate the node resources
        # for the trial
        wait_for_resources = ray.remote(
            num_cpus=resources.cpu_total(),
            num_gpus=resources.gpu_total())(lambda: None)

        error_msg = None
        for _ in range(TRIAL_START_ATTEMPTS):
            try:
                # Wait for ray to scale up the cluster
                wait_status = wait_for_resources.remote()
                ray.get(wait_status, NODE_CREATION_TIMEOUT)

                custom_resources = self._get_node_resources(resources)
                if custom_resources is not None:
                    break
                logger.warning("Trial %s: Not enough nodes, retrying...", trial)
            except RayTimeoutError:
                logger.warning("Trial %s: Timed out, retrying...", trial)
            except Exception as ex:
                logger.exception("Trial %s: Error starting trial, aborting!", trial)
                error_msg = traceback.format_exception(ex)
                break
        else:
            logger.exception(
                "Trial %s: Aborting trial after %s start "
                "attempts!", trial, TRIAL_START_ATTEMPTS)
            self._stop_trial(trial, error=True, error_msg=error_msg)
            return

        # Update trial node affinity configuration and
        # extra_custom_resources requirement
        trial.config.update(__ray_node_affinity__=custom_resources)
        trial.resources.extra_custom_resources.update(custom_resources)
        super().start_trial(trial, checkpoint)

    def has_resources(self, resources):
        if not super().has_resources(resources):
            return False
        elif self._trial_queued:
            # Allowing trial to start even though the cluster does not have
            # enough free resources
            return True

        node_resources = self._get_node_resources(resources)
        if node_resources is not None:
            return True

        if self._queue_trials:
            self._trial_queued = True
            logger.warning(
                "Allowing trial to start even though the "
                "cluster does not have enough free resources. Trial actors "
                "may appear to hang until enough resources are added to the "
                "cluster (e.g., via autoscaling). You can disable this "
                "behavior by specifying `queue_trials=False` in "
                "ray.tune.run().")
            return True
        return False


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
            resource_attr = "gpu"
            world_size = num_gpus
            # Assign one GPU per remote process
            num_gpus = 1
            # Assign extra CPUs for dataloaders
            num_cpus = config.get("workers", 0)
        else:
            resource_attr = "cpu"
            world_size = num_cpus
            # Assign one CPU per remote process
            num_cpus = 1

        # Check for node affinity
        resources_by_node = _get_resources_per_node()
        ray_node_affinity = config.get("__ray_node_affinity__", {})
        node_resource = 0.0
        committed_resource = 0.0
        resources = {}

        # Update the config through SigOpt.
        self.sigopt = None
        if "sigopt_config" in config:
            assert config.get("sigopt_experiment_id", None) is not None
            self.sigopt = SigOptImagenetExperiment(
                experiment_id=config["sigopt_experiment_id"],
                sigopt_config=config["sigopt_config"])
            self.suggestion = self.sigopt.get_next_suggestion()
            self.sigopt.update_config_with_suggestion(config, self.suggestion)
            print("SigOpt suggestion: ", self.suggestion)
            print("Config after Sigopt:")
            pprint(config)
        self.epochs = config["epochs"]

        # Create one ray remote process for each experiment in the process group
        self.procs = []
        for _ in range(world_size):
            if ray_node_affinity and node_resource <= 0:
                node_id, node_resource = ray_node_affinity.popitem()
                proc_per_node = getattr(resources_by_node[node_id], resource_attr, 0)
                committed_resource = 1.0 / proc_per_node
                resources = {node_id: committed_resource}

            experiment = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus,
                                    resources=resources)(ImagenetExperiment)
            self.procs.append(experiment.remote())
            node_resource -= committed_resource

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

        # Save initialized model.
        if config.get("checkpoint_at_init", False):
            self.save()

    def _train(self):
        status = []
        for w in self.procs:
            status.append(w.run_epoch.remote(self.iteration))

        # Wait for remote functions to complete
        results = ray.get(status)

        # Update the sigopt configuration once we're at the end
        if self.iteration >= self.epochs - 1 and self.sigopt is not None:
            if results[0]["mean_accuracy"] > 0.0:
                print("Updating observation with value=", results[0]["mean_accuracy"])
                self.sigopt.update_observation(self.suggestion,
                                               results[0]["mean_accuracy"])
                print("Full results: ")
                pprint(results[0])

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

    # Group trial into nodes as much as possible
    # kwargs.update(trial_executor=AffinityExecutor(
    #     queue_trials=kwargs.get("queue_trials", True),
    #     reuse_actors=kwargs.get("reuse_actors", False),
    #     ray_auto_init=kwargs.get("ray_auto_init", True)
    # ))

    pprint(kwargs)
    tune.run(**kwargs)
    ray.shutdown()
