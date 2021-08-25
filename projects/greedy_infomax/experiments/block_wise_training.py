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

from copy import deepcopy

import numpy as np
import ray.tune as tune
import torch
import torch.nn as nn
import os

from nupic.research.frameworks.greedy_infomax.models.ClassificationModel import (
    ClassificationModel,
)
from nupic.research.frameworks.greedy_infomax.models.UtilityLayers import (
    GradientBlock, EmitEncoding
)
from nupic.research.frameworks.greedy_infomax.mixins.create_block_model import CreateBlockModel
from nupic.research.frameworks.greedy_infomax.utils.model_utils import full_sparse_resnet, small_sparse_resnet

from nupic.research.frameworks.vernon.distributed import mixins, experiments
from nupic.torch.modules import SparseWeights2d
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptExperiment
from nupic.research.frameworks.greedy_infomax.models.BlockModel import BlockModel



from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS
class GreedyInfoMaxExperimentBlockWise(
    experiments.SelfSupervisedExperiment,
):
    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result


DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]

BATCH_SIZE = 32
NUM_EPOCHS = 10

"""
Block wise model example:


model_structure:
[ (model_class=nn.Conv2d, model_args={args}, previous_checkpoint=None, 
save_checkpoint=None, train=True)
(BilinearInfo, {args})
(EmitEncoding)
(GradientBlock)
(PreActBlockNoBN, {args}, previous_checkpoint="checkpoint_file.ckpt", save_checkpoint=None),
(PreActBlockNoBN, {args}, previous_checkpoint="checkpoint_file.ckpt"),
(PreActBlockNoBN, {args}, previous_checkpoint="checkpoint_file.ckpt"),
(PreActBlockNoBN, {args}, previous_checkpoint="checkpoint_file.ckpt"), 
save_checkpoint="save_here.ckpt")
(EmitEncoding)
(BilinearInfo, {args})

]

"""
# model hyperparameters
grayscale=True
num_channels = 64
input_dims = 1
if not grayscale:
    input_dims = 3
block_wise_small_resnet=small_sparse_resnet


BLOCK_WISE_BASE = deepcopy(DEFAULT_BASE)
BLOCK_WISE_BASE.update(dict(
        experiment_class=BlockModel,
        wandb_args=dict(
            project="greedy_infomax-block_wise",
            name=f"block_wise_training",
        ),
        batch_size=32,
        batch_size_supervised=32,
        val_batch_size=32,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        model_class=BlockModel,
        model_args=small_sparse_resnet,
        optimizer_class = torch.optim.SGD,
        optimizer_args=dict(lr=2e-4),
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
                max_lr=0.24, #change based on sparsity/dimensionality
                div_factor=100,  # initial_lr = 0.01
                final_div_factor=1000,  # min_lr = 0.0000025
                pct_start=1.0 / 10.0,
                epochs=10,
                anneal_strategy="linear",
                max_momentum=1e-4,
                cycle_momentum=False,
            ),
        classifier_config=dict(
            model_class=ClassificationModel,
            model_args=dict(num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    ),
)



CONFIGS = dict(

)