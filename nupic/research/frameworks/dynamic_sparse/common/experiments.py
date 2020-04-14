# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from copy import deepcopy

from ray import tune
from ray.tune.suggest.sigopt import SigOptSearch

import nupic.research.frameworks.dynamic_sparse.models as models
import nupic.research.frameworks.dynamic_sparse.networks as networks

from .datasets import load_dataset


class RayTrainable(tune.Trainable):
    """ray.tune trainable generic class. Adaptable to any pytorch module."""

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        network = getattr(networks, config["network"])(config=config)
        self.model = getattr(models, config["model"])(network, config=config)
        self.dataset = load_dataset(config["dataset_name"])(config=config)
        self.model.setup()
        self.experiment_name = config["name"]

    def _train(self):
        log = self.model.run_epoch(self.dataset, self._iteration)
        return log

    def _save(self, checkpoint_dir):
        self.model.save(checkpoint_dir, self.experiment_name)
        return checkpoint_dir

    def _restore(self, checkpoint):
        self.model.restore(checkpoint, self.experiment_name)


class CustomTrainable(tune.Trainable):
    """ray.tune trainable generic class Adaptable to any pytorch module."""

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        model, dataset = config["unpack_params"]()
        self.model = model
        self.dataset = dataset
        print("finished setting up trainable")

    def _train(self):
        print("running epoch")
        log = self.model.run_epoch(self.dataset, self._iteration)
        return log

    def _save(self, checkpoint_dir):
        self.model.save(checkpoint_dir, self.experiment_name)
        return checkpoint_dir

    def _restore(self, checkpoint):
        self.model.restore(checkpoint, self.experiment_name)

    # added method to setup new model
    def setup(self, model, network, dataset):
        self.model = model
        self.network = network
        self.dataset = dataset


def base_experiment(config):
    tune.run(RayTrainable, **config)


def sigopt_experiment(config):
    """
    Requires environment variable SIGOPT_KEY
    """

    exp_config = config["config"]
    if "params_space" not in exp_config:
        raise ValueError("SigOpt experiment require a params_space")
    if "performance_metric" not in exp_config:
        exp_config["performance_metric"] = "val_acc"

    # define algorithm
    algo = SigOptSearch(
        exp_config["params_space"],
        name=config["name"],
        # manually define max concurrent
        max_concurrent=5,
        reward_attr=exp_config["performance_metric"],
        # optimization_budget=100
    )

    tune.run(RayTrainable, search_alg=algo, **config)


def iterative_pruning_experiment(config):
    # get pruning schedule
    if "iterative_pruning_schedule" not in config["config"]:
        raise ValueError(
            """IterativePruningExperiment requires and
            iterative_pruning_schedule to be defined"""
        )
    else:
        pruning_schedule = config["config"]["iterative_pruning_schedule"]

    # first run saves its initial weights and final weights
    instance_config = deepcopy(config)
    instance_config["config"]["target_final_density"] = 1.0
    instance_config["config"]["first_run"] = True
    tune.run(RayTrainable, **instance_config)

    # ensures pruning_schedule is reversed
    pruning_schedule = sorted(pruning_schedule, reverse=True)
    # ensures 1.0 density is not included
    if pruning_schedule[0] >= 1.0:
        pruning_schedule = pruning_schedule[1:]
    # no longer need initial weights
    instance_config["config"]["first_run"] = False
    for target_density in pruning_schedule:
        instance_config["config"]["target_final_density"] = target_density
        tune.run(RayTrainable, **instance_config)
