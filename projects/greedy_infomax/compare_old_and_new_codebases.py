# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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


import torch

from experiments import CONFIGS
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    all_module_multiple_log_softmax,
)

if __name__ == "__main__":
    experiment_config = CONFIGS["small_block_flatten"]
    experiment_config["distributed"] = False
    experiment_config["batches_in_epoch"] = 1
    experiment_config["batches_in_epoch_val"] = 1
    experiment_config["batches_in_epoch_supervised"] = 1
    experiment_config["supervised_training_epochs_per_validation"] = 0
    experiment_config["batch_size"]=2
    experiment_class = experiment_config["experiment_class"]()
    experiment_class.setup_experiment(experiment_config)
    experiment_class.run_epoch()