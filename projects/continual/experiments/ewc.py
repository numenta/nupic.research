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
from nupic.research.frameworks.pytorch.models import EWCNetwork, StandardMLP
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins
from nupic.research.frameworks.vernon.run_experiment.trainables import (
    ContinualLearningTrainable,
)


class EWCContinualLearningExperiment(mixins.ElasticWeightConsolidation,
                                     ContinualLearningExperiment):
    pass


class ReduceLRContinualLearningExperiment(mixins.ReduceLRAfterTask,
                                          ContinualLearningExperiment):
    pass


# Regular continual learning, no EWC, 5 tasks
# avg acc .499 -> acc per task [.398, .361, .546, .839, .364]

# Same experiment, with EWC
# avg acc .529 -> acc per task [.514, .437, .584, .836, .278]

# Same experiment, reducing learning rate after first task, no EWC
# sanity check
# avg acc .2339 -> acc per task [.976, .006, .005, .002, 0.000]


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
    optimizer_args=dict(lr=5e-3),
)

cl_mnist_ewc = deepcopy(cl_mnist)
cl_mnist_ewc.update(
    experiment_class=EWCContinualLearningExperiment,
    ewc_lambda=400,
)

cl_mnist_reduce = deepcopy(cl_mnist)
cl_mnist_reduce.update(
    experiment_class=ReduceLRContinualLearningExperiment,
    new_lr=2e-3,
)

# Mimics the original implementation in the repo
# useful to compare experiments.
ewc_repr = deepcopy(cl_mnist)
ewc_repr.update(
    experiment_class=EWCContinualLearningExperiment,
    epochs=3,
    num_classes=10,
    num_tasks=10,
    ewc_lambda=40,
    optimizer_class=torch.optim.SGD,
    optimizer_args=dict(lr=1e-1, weight_decay=0),
    batch_size=128,
    ewc_fisher_sample_size=1024,
    model_class=EWCNetwork,
    model_args=dict(input_size=28 * 28, output_size=10),
)

# Export configurations in this file
CONFIGS = dict(
    cl_mnist=cl_mnist,
    cl_mnist_ewc=cl_mnist_ewc,
    cl_mnist_reduce=cl_mnist_reduce,
    ewc_repr=ewc_repr
)
