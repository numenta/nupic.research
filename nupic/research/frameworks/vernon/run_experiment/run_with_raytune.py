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
from nupic.research.frameworks.vernon.run_experiment.search import TrialsCollection
from nupic.research.support.ray_utils import (
    get_last_checkpoint,
    register_torch_serializers,
)

from .trainables import BaseTrainable, SigOptImagenetTrainable, SupervisedTrainable

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def run(config):

    if config.get("single_instance", False):
        return run_single_instance(config)

    # Connect to ray
    address = os.environ.get("REDIS_ADDRESS", config.get("redis_address"))
    ray.init(address=address, local_mode=config.get("local_mode", False))

    # Register serializer and deserializer - needed when logging arrays and tensors.
    register_torch_serializers()

    # Build kwargs for `tune.run` function using merged config and command line dict
    kwargs_names = tune.run.__code__.co_varnames[:tune.run.__code__.co_argcount]

    if "sigopt_config" in config:
        kwargs = dict(zip(kwargs_names, [SigOptImagenetTrainable,
                                         *tune.run.__defaults__]))
    else:
        ray_trainable = config.get("ray_trainable", SupervisedTrainable)
        assert issubclass(ray_trainable, BaseTrainable)
        kwargs = dict(zip(kwargs_names, [ray_trainable, *tune.run.__defaults__]))

    # Check if restoring experiment from last known checkpoint
    if config.pop("restore", False):
        result_dir = os.path.join(config["local_dir"], config["name"])
        config["restore_checkpoint_file"] = get_last_checkpoint(result_dir)

    # Update`tune.run` kwargs with config
    kwargs.update(config)
    kwargs["config"] = config

    # Make sure to only select`tune.run` function arguments
    kwargs = dict(filter(lambda x: x[0] in kwargs_names, kwargs.items()))

    # Queue trials until the cluster scales up
    kwargs.update(queue_trials=True)

    pprint(kwargs)
    result = tune.run(**kwargs)
    ray.shutdown()
    return result


def run_single_instance(config):

    config.setdefault("num_gpus", torch.cuda.device_count())
    config["workers"] = config.get("workers", 4)
    config["log_level"] = "INFO"
    config["reuse_actors"] = False
    config["dist_port"] = get_free_port()

    ray_trainable = config.get("ray_trainable", SupervisedTrainable)
    assert issubclass(ray_trainable, Trainable)

    # Build kwargs for `tune.run` function using merged config and command line dict
    kwargs_names = tune.run.__code__.co_varnames[:tune.run.__code__.co_argcount]
    kwargs = dict(zip(kwargs_names, [ray_trainable, *tune.run.__defaults__]))
    # Update`tune.run` kwargs with config
    kwargs.update(config)
    kwargs["config"] = config

    # Update tune stop criteria with config epochs
    stop = kwargs.get("stop", {}) or dict()

    stop_condition = getattr(ray_trainable, "stop_condition", "epochs")
    stop_iteration = config.get(stop_condition, None)
    if stop_iteration:
        stop.update(training_iteration=stop_iteration)

    kwargs["stop"] = stop
    # Make sure to only select`tune.run` function arguments
    kwargs = dict(filter(lambda x: x[0] in kwargs_names, kwargs.items()))
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
    print(config)
    tune.run(**kwargs)
    print("**** Trial ended")
