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
from functools import reduce

import torch
import torch.nn as nn
from torch.nn import init

from nupic.research.frameworks.continual_learning import experiments, mixins
from nupic.research.frameworks.pytorch.datasets import torchvisiondataset
from nupic.research.frameworks.pytorch.models import StandardMLP


class EWCContinualLearningExperiment(mixins.ElasticWeightConsolidation,
                                     experiments.ContinualLearningExperiment):
    pass


class ReduceLRContinualLearningExperiment(mixins.ReduceLRAfterTask,
                                          experiments.ContinualLearningExperiment):
    pass


class EWCNetwork(nn.Module):

    def __init__(self, input_size, output_size,
                 hidden_size=400,
                 hidden_layer_num=2,
                 hidden_dropout_prob=.5,
                 input_dropout_prob=.2,
                 ):
        """
        Adapted nearly verbatim from
        https://github.com/kuc2477/pytorch-ewc/blob/master/model.py

        Used to test consistency in the algorithm implementation.
        """

        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size

        # Layers.
        self.layers = nn.ModuleList([
            # input
            nn.Flatten(),
            nn.Linear(self.input_size, self.hidden_size), nn.ReLU(),
            nn.Dropout(self.input_dropout_prob),
            # hidden
            *((nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
               nn.Dropout(self.hidden_dropout_prob)) * self.hidden_layer_num),
            # output
            nn.Linear(self.hidden_size, self.output_size)
        ])

        self.xavier_initialize()

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)

    def xavier_initialize(self):
        modules = [
            m for n, m in self.named_modules() if
            "conv" in n or "linear" in n
        ]

        parameters = [
            p for
            m in modules for
            p in m.parameters() if
            p.dim() >= 2
        ]

        for p in parameters:
            init.xavier_normal(p)


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
    experiment_class=experiments.ContinualLearningExperiment,
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
