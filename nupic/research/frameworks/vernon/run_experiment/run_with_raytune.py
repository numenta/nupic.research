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

import os
import time
from pprint import pprint

import ray
import ray.resource_spec
import torch
from ray.tune import Trainable, tune

from nupic.research.frameworks.vernon.experiment_utils import get_free_port
from nupic.research.frameworks.vernon.search import TrialsCollection
from nupic.research.support.ray_utils import (
    get_last_checkpoint,
    register_torch_serializers,
)

from .trainables import SigOptImagenetTrainable, SupervisedTrainable

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def run(config):

    if config.get("single_instance", False):
        return run_single_instance(config)

    # Connect to ray
    address = os.environ.get("REDIS_ADDRESS", config.get("redis_address"))
    ray.init(address=address, local_mode=config.get("local_mode", False))

    # Register serializer and deserializer - needed when logging arrays and tensors.
    register_torch_serializers()

    # Get ray.tune kwargs for the given config.
    kwargs = get_tune_kwargs(config)

    # Queue trials until the cluster scales up
    kwargs.update(queue_trials=True)

    pprint(kwargs)
    result = tune.run(**kwargs)
    ray.shutdown()
    return result


def run_single_instance(config):

    # Get number of GPUs
    config.setdefault("num_gpus", torch.cuda.device_count())
    config["workers"] = 4
    config["log_level"] = "INFO"
    config["reuse_actors"] = False
    config["dist_port"] = get_free_port()

    # Get ray.tune kwargs for the given config.
    kwargs = get_tune_kwargs(config)
    pprint(kwargs)

    # Only run trial collection if specifically requested
    if config.get("use_trial_collection", False):
        # Current torch distributed approach requires num_samples to be 1
        num_samples = 1
        if "num_samples" in kwargs:
            num_samples = kwargs["num_samples"]
            kwargs["num_samples"] = 1

        trials = TrialsCollection(kwargs["config"], num_samples, restore=True)
        t_init = time.time()

        for config in trials.retrieve():
            t0 = time.time()
            trials.report_progress()
            run_trial_single_instance(config, kwargs)
            # Report time elapsed
            t1 = time.time()
            print(f"***** Time elapsed last trial: {t1-t0:.0f} seconds")
            print(f"***** Time elapsed total: {t1-t_init:.0f} seconds")
            # Save trials for later retrieval
            ray.shutdown()
            trials.mark_completed(config, save=True)

        print(f"***** Experiment {trials.name} finished: {len(trials.completed)}"
              " trials completed")
    else:
        run_trial_single_instance(config, kwargs),
        ray.shutdown()


def run_trial_single_instance(config, kwargs):
    # Connect to ray, no specific redis address
    ray.init(load_code_from_local=False, webui_host="0.0.0.0")
    config["dist_url"] = f"tcp://127.0.0.1:{get_free_port()}"
    kwargs["config"] = config
    tune.run(**kwargs)
    print("**** Trial ended")


def get_tune_kwargs(config):
    """
    Build and return the kwargs needed to run `tune.run` for a given config.

    :param config:
        - ray_trainable: the ray.tune.Trainable; defaults to SupervisedTrainable
            - stop_condition: If the trainable has this attribute, it will be used
                              to decide which config parameter dictates the stop
                              training_iteration.
        - sigopt_config: (optional) used for running experiments with SigOpt and
                         the SigOptImagenetTrainable
        - restore: whether to restore from the latest checkpoint; defaults to False
        - local_dir: needed with 'restore'; identifies the parent directory of
                     experiment results.
        - name: needed with 'restore'; local_dir/name identifies the path to
                the experiment checkpoints.
    """

    # Build kwargs for `tune.run` function using merged config and command line dict
    kwargs_names = tune.run.__code__.co_varnames[:tune.run.__code__.co_argcount]

    # Zip the kwargs along with the Ray trainable.
    if "sigopt_config" in config:
        kwargs = dict(zip(kwargs_names, [SigOptImagenetTrainable,
                                         *tune.run.__defaults__]))
    else:
        ray_trainable = config.get("ray_trainable", SupervisedTrainable)
        assert issubclass(ray_trainable, Trainable)
        kwargs = dict(zip(kwargs_names, [ray_trainable, *tune.run.__defaults__]))

    # Check if restoring experiment from last known checkpoint
    if config.pop("restore", False):
        result_dir = os.path.join(config["local_dir"], config["name"])
        config["restore_checkpoint_file"] = get_last_checkpoint(result_dir)

    # Update`tune.run` kwargs with config
    kwargs.update(config)
    kwargs["config"] = config

    # Make sure to only select `tune.run` function arguments
    kwargs = dict(filter(lambda x: x[0] in kwargs_names, kwargs.items()))

    # Collect tune stop criteria.
    stop = kwargs.get("stop", {}) or dict()

    # Update the stop criteria `training_iteration` with the named `stop_condition`
    # This may be `epochs` or `num_tasks`, for instance.
    stop_condition = getattr(ray_trainable, "stop_condition", "epochs")
    stop_iteration = config.get(stop_condition, None)
    if stop_iteration is not None:
        stop.update(training_iteration=stop_iteration)

    # Update the stop condition.
    kwargs["stop"] = stop

    return kwargs
