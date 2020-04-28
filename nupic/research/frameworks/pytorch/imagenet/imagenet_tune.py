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
import time
from pprint import pprint

import ray
import ray.resource_spec
import ray.util.sgd.utils as ray_utils
import torch
from ray.exceptions import RayActorError
from ray.tune import Trainable, tune
from ray.tune.resources import Resources
from ray.tune.utils import warn_if_slow

from nupic.research.frameworks.pytorch.imagenet import ImagenetExperiment
from nupic.research.frameworks.pytorch.imagenet.experiment_search import (
    TrialsCollection,
)
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import get_free_port
from nupic.research.frameworks.sigopt import SigOptImagenetExperiment

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
        # Configure logging related stuff
        log_format = config.get("log_format", logging.BASIC_FORMAT)
        log_level = getattr(logging, config.get("log_level", "INFO").upper())
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger(config.get("name", type(self).__name__))
        self.logger.setLevel(log_level)
        self.logger.addHandler(console)

        self.logger.debug(
            f"_setup: trial={self._trial_info.trial_name}({self.iteration}), "
            f"config={config}")

        # Try to recover a trial at least this many times
        self.max_retries = max(config.get("max_retries", 3), 0)

        # Create ray remote workers
        self._create_workers(config)

        # Save initialized model
        if config.get("checkpoint_at_init", False):
            self.save()

    def _train(self):
        self.logger.debug(f"_train: {self._trial_info.trial_name}({self.iteration})")
        try:
            status = []
            for w in self.procs:
                status.append(w.run_epoch.remote(self.iteration))

            # Wait for remote functions and check for errors
            # Return the results from the first remote function
            if ray_utils.check_for_failure(status):
                results = ray.get(status)
                ret = copy.deepcopy(results[0])
                self._process_result(ret)
                return ret

            err_msg = (f"{self._trial_info.trial_name}({self.iteration}): "
                       f"One of the remote workers failed during training")
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        except Exception:
            self._kill_workers()
            raise

    def _save(self, _=None):
        self.logger.debug(f"_save: {self._trial_info.trial_name}({self.iteration})")
        # All models are synchronized. Just save the state of first model
        with warn_if_slow("ImagenetExperiment.get_state.remote"):
            return ray.get(self.procs[0].get_state.remote())

    def _restore(self, state):
        self.logger.debug(f"_restore: {self._trial_info.trial_name}({self.iteration})")
        # Restore the state to every process
        state_id = ray.put(state)
        ray.get([w.set_state.remote(state_id) for w in self.procs])

    def _create_workers(self, config):
        """
        Create one ray remote process for each GPU/process
        """
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

        self._process_config(config)

        for i in range(1 + self.max_retries):
            self.procs = []
            for _ in range(world_size):
                experiment = ray.remote(
                    num_cpus=num_cpus, num_gpus=num_gpus)(ImagenetExperiment)
                self.procs.append(experiment.remote())

            # Use first process as head of the group
            ip = ray.get(self.procs[0].get_node_ip.remote())
            port = ray.get(self.procs[0].get_free_port.remote())
            port = config.get("dist_port", port)
            dist_url = "tcp://{}:{}".format(ip, port)

            # Configure each process in the group
            status = []
            for rank, w in enumerate(self.procs):
                worker_config = copy.deepcopy(config)
                worker_config["distributed"] = True
                worker_config["dist_url"] = dist_url
                worker_config["world_size"] = world_size
                worker_config["rank"] = rank
                status.append(w.setup_experiment.remote(worker_config))
                self.logger.debug(
                    f"_create_workers: rank={rank}, "
                    f"trial={self._trial_info.trial_name}({self.iteration})")

            # Wait for remote function and check for errors
            if ray_utils.check_for_failure(status):
                return

            # Remote function failed, kill workers and try again
            self.logger.warning(f"Failed to create workers, "
                                f"retrying {i + 1}/{self.max_retries}")
            # Restart all workers on failure
            self._kill_workers()

            # Back off a few seconds
            time.sleep(2 ** i)
        else:
            # Reached max failures
            err_msg = f"Failed to create workers after {self.max_retries} retries"
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

    def _kill_workers(self):
        for w in self.procs:
            self.logger.warning(
                f"Killing worker {w}, "
                f"trial={self._trial_info.trial_name}({self.iteration})")
            ray.kill(w)
        self.procs = []

    def _stop(self):
        self.logger.debug(f"_stop: {self._trial_info.trial_name}({self.iteration})")
        try:
            status = [w.stop_experiment.remote() for w in self.procs]
            # wait until all remote workers stop
            ray.get(status)
            for w in self.procs:
                w.__ray_terminate__.remote()
            self.procs = []
        except RayActorError as ex:
            self.logger.warning("Failed to shutdown gracefully", exc_info=ex)
            self._kill_workers()

    def _process_config(self, config):
        pass

    def _process_result(self, result):
        pass


class SigOptImagenetTrainable(ImagenetTrainable):
    """
    This class updates the config using SigOpt before the models and workers are
    instantiated, and updates the result using SigOpt once training completes.
    """

    def _process_config(self, config):
        """
        :param config:
            Dictionary configuration of the trainable

            - sigopt_experiment_id: id of experiment
            - sigopt_config: dict to specify configuration of sigopt experiment
            - sigopt_experiment_class: class inherited from `SigoptExperiment` which
                                       characterizes how the trainable will get and
                                       utilize suggestions

        """
        # Update the config through SigOpt.
        self.sigopt = None
        if "sigopt_config" in config:
            assert config.get("sigopt_experiment_id", None) is not None

            # Check for user specified sigopt-experiment class.
            experiment_class = config.get(
                "sigopt_experiment_class", SigOptImagenetExperiment)

            # Instantiate experiment.
            self.sigopt = experiment_class(
                experiment_id=config["sigopt_experiment_id"],
                sigopt_config=config["sigopt_config"])

            # Get suggestion and update config.
            self.suggestion = self.sigopt.get_next_suggestion()
            self.sigopt.update_config_with_suggestion(config, self.suggestion)
            print("SigOpt suggestion: ", self.suggestion)
            print("Config after Sigopt:")
            pprint(config)
            self.epochs = config["epochs"]

            # Get names of performance metrics.
            assert "metrics" in config["sigopt_config"]
            self.metric_names = [
                metric["name"] for metric in config["sigopt_config"]["metrics"]
            ]
            assert "mean_accuracy" in self.metric_names, \
                "For now, we only update the observation if `mean_accuracy` is present."

    def _process_result(self, result):
        # Update sigopt with the new result once we're at the end
        if self.sigopt is not None:
            result["early_stop"] = result.get("early_stop", 0.0)
            if self.iteration >= self.epochs - 1:
                result["early_stop"] = 1.0
                if result["mean_accuracy"] > 0.0:
                    print("Updating observation with value=", result["mean_accuracy"])

                    # Collect and report relevant metrics.
                    values = [
                        dict(name=name, value=result[name])
                        for name in self.metric_names
                    ]
                    self.sigopt.update_observation(self.suggestion, values=values)
                    print("Full results: ")
                    pprint(result)


def run(config):
    # Connect to ray
    address = os.environ.get("REDIS_ADDRESS", config.get("redis_address"))
    ray.init(address=address)

    # Build kwargs for `tune.run` function using merged config and command line dict
    kwargs_names = tune.run.__code__.co_varnames[:tune.run.__code__.co_argcount]

    if "sigopt_config" in config:
        kwargs = dict(zip(kwargs_names, [SigOptImagenetTrainable,
                                         *tune.run.__defaults__]))
    else:
        imagenet_trainable = config.get("imagenet_trainable", ImagenetTrainable)
        assert issubclass(imagenet_trainable, ImagenetTrainable)
        kwargs = dict(zip(kwargs_names, [imagenet_trainable, *tune.run.__defaults__]))

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


def run_single_instance(config):

    # get number of GPUs
    config["num_gpus"] = torch.cuda.device_count()
    config["workers"] = 4
    config["log_level"] = "INFO"
    config["reuse_actors"] = False

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
    # pprint(kwargs)

    # current torch distributed approach requires num_samples to be 1
    num_samples = 1
    if "num_samples" in kwargs:
        num_samples = kwargs["num_samples"]
        kwargs["num_samples"] = 1

    trials = TrialsCollection(kwargs["config"], num_samples, restore=True)
    t_init = time.time()

    for config in trials.retrieve():
        t0 = time.time()
        trials.report_progress()

        # Connect to ray, no specific redis address
        ray.init(load_code_from_local=True, webui_host="0.0.0.0")

        config["dist_url"] = f"tcp://127.0.0.1:{get_free_port()}"
        kwargs["config"] = config
        tune.run(**kwargs)
        print("**** ended training")

        # report time elapsed
        t1 = time.time()
        print(f"***** Time elapsed last trial: {t1-t0:.0f} seconds")
        print(f"***** Time elapsed total: {t1-t_init:.0f} seconds")

        ray.shutdown()

        # save trials for later retrieval
        trials.mark_completed(config, save=True)

        # sleep to avoid interference between runs
        time.sleep(2)

        # error message when experiment ends

    print(f"***** Experiment {trials.name} finished: {len(trials.completed)}"
          " trials completed")
