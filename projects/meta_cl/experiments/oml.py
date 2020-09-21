#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import copy
import os

import torch

from nupic.research.frameworks.pytorch.datasets import omniglot
from nupic.research.frameworks.pytorch.models import OMLNetwork
from nupic.research.frameworks.vernon import MetaContinualLearningExperiment, mixins
from nupic.research.frameworks.vernon.run_experiment.trainables import (
    SupervisedTrainable,
)


class OMLExperiment(mixins.OnlineMetaLearning,
                    MetaContinualLearningExperiment):
    pass


# Alternative to run on a single GPU
def run_experiment(config):
    exp = config.get("experiment_class")()
    exp.setup_experiment(config)
    print(f"Training started....")
    while not exp.should_stop():
        result = exp.run_epoch()
        print(f"Accuracy: {result['mean_accuracy']:.4f}")
    print(f"....Training finished")


# Simple Omniglot Experiment
metacl_test = dict(
    # training infrastructure
    ray_trainable=SupervisedTrainable,
    distributed=False,
    # problem specific
    experiment_class=MetaContinualLearningExperiment,
    dataset_class=omniglot,
    model_class=OMLNetwork,
    model_args=dict(input_size=(105, 105)),
    # metacl variables
    num_classes=50,
    batch_size=5,
    val_batch_size=15,
    slow_batch_size=64,
    replay_batch_size=64,
    epochs=1000,
    tasks_per_epoch=10,
    adaptation_lr=0.03,
    # generic
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=1e-4),
    dataset_args=dict(root="~/nta/datasets"),
    local_dir=os.path.expanduser("~/nta/results/experiments/meta_cl"),
)

metacl_oml = copy.deepcopy(metacl_test)
metacl_oml.update(
    experiment_class=OMLExperiment,
    run_meta_test=True,
)

# Export configurations in this file
CONFIGS = dict(
    metacl_test=metacl_test,
    metacl_oml=metacl_oml,
)
