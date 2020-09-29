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

"""
Base GSC Experiment configuration.
"""

import os
from copy import deepcopy

import torch

from nupic.research.frameworks.pytorch.datasets import torchvisiondataset
from nupic.research.frameworks.pytorch.models import StandardMLP
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.research.frameworks.vernon.run_experiment.trainables import (
    ContinualLearningTrainable
)


class EWCContinualLearningExperiment(mixins.ElasticWeightConsolidation,
                                            ContinualLearningExperiment):
    pass


class ReduceLRContinualLearningExperiment(mixins.ReduceLRAfterTask,
                                          ContinualLearningExperiment):
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


cl_mnist = dict(
    local_dir=os.path.expanduser("~/nta/results/experiments/meta_cl"),
    seed=123,
    # specific to continual learning
    distributed=False,
    evaluation_metrics=[
        "eval_all_visited_tasks",
        "eval_individual_tasks",
    ],
    ray_trainable=ContinualLearningTrainable,
    experiment_class=ContinualLearningExperiment,
    num_classes=10,
    num_tasks=5,
    # regular experiments
    dataset_class=torchvisiondataset,
    model_class=StandardMLP,
    model_args=dict(input_size=(28, 28), num_classes=10),
    dataset_args=dict(root="~/nta/datasets", dataset_name="MNIST"),
    # epochs
    epochs=2,
    batch_size=1024,
    # optimizer
    optimizer_class=torch.optim.SGD,
    # .1828 -> [0, 0, 0, 0, .92]
    optimizer_args=dict(lr=1e-2),
    # .4995 -> [.39, .36, .54, .83, .36]
    # optimizer_args=dict(lr=5e-3),
)

cl_mnist_ewc = deepcopy(cl_mnist)
cl_mnist_ewc.update(
    experiment_class=EWCContinualLearningExperiment,
    optimizer_args=dict(lr=5e-3),
    # .5032 -> [.40, .37, .55, .84, .34]
    # ewc_lambda=40,
    # .5041 -> [.40, .38, .56, .84, .33]
    # ewc_lambda=80,
    # .5155 -> [.44, .41, .58, .84, .29]
    # ewc_lambda=400,
    # .529 -> [.51, .43, .58, .83, .27]
    ewc_lambda=800,
)

# there is evidence it is working, it is doing what it is supposed to be
# might just be a matter of getting the right hyperparameters

# there is a limit to how high lambda can be
# the actual question here is not whether this is the best results possible -
# its just within our framework, can we replicate the results from a known repo?

# .5978 -> [.94, .35, .57, .75, .34]
cl_mnist_reduce = deepcopy(cl_mnist)
cl_mnist_reduce.update(
    experiment_class=ReduceLRContinualLearningExperiment,
    new_lr=5e-3,
)

# Export configurations in this file
CONFIGS = dict(
    cl_mnist=cl_mnist,
    cl_mnist_ewc=cl_mnist_ewc,
    cl_mnist_reduce=cl_mnist_reduce
)


