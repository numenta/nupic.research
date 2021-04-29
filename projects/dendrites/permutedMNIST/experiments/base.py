#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
Base Experiment configuration.
"""

import numpy as np
import torch
import torch.nn.functional as F

from nupic.research.frameworks.dendrites import DendriticMLP
from nupic.research.frameworks.pytorch.datasets import ContextDependentPermutedMNIST
from nupic.research.frameworks.vernon import ContinualLearningExperiment, mixins


class PermutedMNISTExperiment(mixins.RezeroWeights,
                              ContinualLearningExperiment):
    pass


NUM_TASKS = 2

DEFAULT_BASE = dict(
    experiment_class=PermutedMNISTExperiment,

    dataset_class=ContextDependentPermutedMNIST,
    dataset_args=dict(
        num_tasks=NUM_TASKS,
        dim_context=1024,
        seed=np.random.randint(0, 1000),
        download=True,  # Change to True if running for the first time
    ),

    model_class=DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[64, 64],
        num_segments=NUM_TASKS,
        dim_context=1024,  # Note: with the Gaussian dataset, `dim_context` was
        # 2048, but this shouldn't effect results
        kw=True,
        # dendrite_sparsity=0.0,
    ),

    batch_size=256,
    val_batch_size=512,
    epochs=1,
    epochs_to_validate=(4, 9, 24, 49),  # Note: `epochs_to_validate` is treated as
    # the set of task ids after which to
    # evaluate the model on all seen tasks
    num_tasks=NUM_TASKS,
    num_classes=10 * NUM_TASKS,
    distributed=False,
    seed=np.random.randint(0, 10000),

    loss_function=F.nll_loss,
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=0.001),
)


# Export configurations in this file
CONFIGS = dict(
    default_base=DEFAULT_BASE,
)
