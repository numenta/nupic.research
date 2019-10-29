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

import nupic.research.frameworks.dynamic_sparse.models as models
import nupic.research.frameworks.dynamic_sparse.networks as networks
from .datasets import Dataset

class RayTrainable(tune.Trainable):
    """ray.tune trainable generic class Adaptable to any pytorch module."""

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        network = getattr(networks, config["network"])(config=config)
        self.model = getattr(models, config["model"])(network, config=config)
        self.dataset = Dataset(config=config)
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


class BaseExperiment():

    def __init__(self, config):
         tune.run(RayTrainable, **config)

class IterativePruningExperiment():

    def __init__(self, config):
        # get pruning schedule
        if 'iterative_pruning_schedule' not in config['config']:
            raise ValueError("IterativePruningExperiment requires a iterative_pruning_schedule to be defined")
        else:
            pruning_schedule = config['config']['iterative_pruning_schedule']

        # first run saves its initial weights and final weights
        instance_config = deepcopy(config)
        instance_config['config']['target_final_density'] = 1.0
        instance_config['config']['first_run'] = True        
        tune.run(RayTrainable, **instance_config)

        # ensures pruning_schedule is reversed
        pruning_schedule = sorted(pruning_schedule, reverse=True)
        # ensures 1.0 density is not included
        if pruning_schedule[0] >= 1.0:
            pruning_schedule = pruning_schedule[1:]
        # no longer need initial weights
        instance_config['config']['first_run'] = False                
        for target_density in pruning_schedule:
            instance_config['config']['target_final_density'] = target_density
            tune.run(RayTrainable, **instance_config)

