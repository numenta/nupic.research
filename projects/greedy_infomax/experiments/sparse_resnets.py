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

import ray.tune as tune
import torch

from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    Classifier,
)
from nupic.research.frameworks.greedy_infomax.models.full_model import (
    SparseFullVisionModel,
)
from nupic.research.frameworks.vernon.distributed import experiments, mixins
from nupic.torch.modules import SparseWeights2d

from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS


class GreedyInfoMaxExperimentSparse(
    mixins.LogEveryLoss,
    mixins.RezeroWeights,
    mixins.LogBackpropStructure,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result


DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]

BATCH_SIZE = 32
NUM_EPOCHS = 10
SPARSE_BASE = deepcopy(DEFAULT_BASE)
model_args = DEFAULT_BASE["model_args"]
static_sparse_weights = dict(
    # weight sparsity
    sparsity=dict(
        conv1=0.01,  # dense
        encoder1=dict(
            block1=dict(conv1=0.7, conv2=0.7),
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.1,  # dense weights
        ),
        encoder2=dict(
            block1=dict(conv1=0.7, conv2=0.7, shortcut=0.01),
            block2=dict(conv1=0.8, conv2=0.8),
            block3=dict(conv1=0.8, conv2=0.8),
            block4=dict(conv1=0.8, conv2=0.8),
            bilinear_info=0.01,
        ),
        encoder3=dict(
            block1=dict(conv1=0.8, conv2=0.8, shortcut=0.01),  # dense
            block2=dict(conv1=0.8, conv2=0.8),
            block3=dict(conv1=0.8, conv2=0.8),
            block4=dict(conv1=0.8, conv2=0.8),
            block5=dict(conv1=0.8, conv2=0.8),
            block6=dict(conv1=0.8, conv2=0.8),
            bilinear_info=0.01,  # dense
        ),
    )
)
static_sparse_activations = dict(
    percent_on=dict(
        encoder1=dict(
            block1=dict(
                nonlinearity1=1.0, nonlinearity2=1.0  # dense, num_channels = 64
            ),
            block2=dict(nonlinearity1=1.0, nonlinearity2=1.0),
            block3=dict(nonlinearity1=1.0, nonlinearity2=1.0),
        ),
        encoder2=dict(
            block1=dict(nonlinearity1=1.0, nonlinearity2=1.0),
            block2=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block3=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block4=dict(nonlinearity1=0.3, nonlinearity2=0.3),
        ),
        encoder3=dict(
            block1=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block2=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block3=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block4=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block5=dict(nonlinearity1=0.3, nonlinearity2=0.3),
            block6=dict(nonlinearity1=0.3, nonlinearity2=0.3),
        ),
    )
)

model_args.update(dict(sparse_weights_class=SparseWeights2d))
sparse_weights_only_args = deepcopy(model_args)
sparse_weights_only_args.update(static_sparse_weights)
sparse_activations_only_args = deepcopy(model_args)
sparse_activations_only_args.update(static_sparse_activations)
sparse_weights_and_activations_args = deepcopy(sparse_weights_only_args)
sparse_weights_and_activations_args.update(static_sparse_activations)

SPARSE_BASE.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="sparse_resnet_base"
        ),
        experiment_class=GreedyInfoMaxExperimentSparse,
        epochs=NUM_EPOCHS,
        epochs_to_validate=range(NUM_EPOCHS),
        supervised_training_epochs_per_validation=10,
        batch_size=BATCH_SIZE,
        model_class=SparseFullVisionModel,
        model_args=model_args,
    )
)
STATIC_SPARSE_WEIGHTS_ONLY = deepcopy(SPARSE_BASE)
STATIC_SPARSE_WEIGHTS_ONLY.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="sparse_weights_only"
        ),
        model_args=sparse_weights_only_args,
    )
)
STATIC_SPARSE_ACTIVATIONS_ONLY = deepcopy(SPARSE_BASE)
STATIC_SPARSE_ACTIVATIONS_ONLY.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="sparse_activations_only"
        ),
        model_args=sparse_activations_only_args,
    )
)
STATIC_SPARSE_WEIGHTS_AND_ACTIVATIONS = deepcopy(SPARSE_BASE)
STATIC_SPARSE_WEIGHTS_AND_ACTIVATIONS.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity",
            name="sparse_weights_and_activations",
        ),
        model_args=sparse_weights_and_activations_args,
    )
)


static_sparse_first_layer_grid_search = deepcopy(static_sparse_weights)
static_sparse_first_layer_grid_search["sparsity"]["conv1"] = tune.grid_search(
    [0.0, 0.25, 0.5, 0.75, 0.85, 0.95]
)
STATIC_SPARSE_FIRST_LAYER_GRID_SEARCH = deepcopy(SPARSE_BASE)
STATIC_SPARSE_FIRST_LAYER_GRID_SEARCH.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="sparse_weights_grid_search"
        ),
        model_args=static_sparse_first_layer_grid_search,
        epochs=5,
    )
)
full_weight_sparsity = deepcopy(static_sparse_weights)
full_weight_sparsity["sparsity"]["conv1"] = 1.0
STATIC_SPARSE_FIRST_LAYER_FULL_SPARSITY = deepcopy(SPARSE_BASE)
STATIC_SPARSE_FIRST_LAYER_FULL_SPARSITY.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="full_sparse_conv1"
        ),
        model_args=full_weight_sparsity,
        epochs=5,
    )
)


LARGE_SPARSE_WEIGHTS_ACTIVATIONS = deepcopy(SPARSE_BASE)
NUM_CLASSES = 10
LARGE_SPARSE_WEIGHTS_ACTIVATIONS.update(
    dict(
        wandb_args=dict(project="greedy_infomax-sparsity-tests", name="large_sparse"),
        experiment_class=GreedyInfoMaxExperimentSparse,
        epochs=NUM_EPOCHS,
        epochs_to_validate=[NUM_EPOCHS - 1],
        batch_size=16,
        supervised_training_epochs_per_validation=10,
        model_class=SparseFullVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
            block_dims=[3, 4, 6],
            num_channels=[512, 512, 512],
            sparse_weights_class=SparseWeights2d,
            sparsity=sparse_weights_only_args["sparsity"],
            percent_on=sparse_activations_only_args["percent_on"],
        ),
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(in_channels=512, num_classes=NUM_CLASSES),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)


# LARGE_SPARSE_GRID_SEARCH = deepcopy(LARGE_SPARSE)
# LARGE_SPARSE_GRID_SEARCH.update(
#     dict(
#         wandb_args=dict(project="greedy_infomax-large_sparse", name="gridsearch"),
#         experiment_class=GreedyInfoMaxExperimentSparse,
#         epochs=NUM_EPOCHS,
#         epochs_to_validate=[NUM_EPOCHS - 1],
#         batch_size=16,
#         supervised_training_epochs_per_validation=10,
#         model_class=VDropSparseFullVisionModel,
#         model_args=dict(
#             negative_samples=16,
#             k_predictions=5,
#             resnet_50=False,
#             grayscale=True,
#             patch_size=16,
#             overlap=2,
#             block_dims=[3, 4, 6],
#             num_channels=tune.grid_search(
#                 [
#                     [64, 128, 256],
#                     [128, 128, 256],
#                     [128, 256, 256],
#                     [256, 256, 256],
#                     [256, 512, 256],
#                     [512, 512, 256],
#                 ]
#             ),
#             sparse_weights_class=SparseWeights2d,
#         ),
#         classifier_config=dict(
#             model_class=Classifier,
#             model_args=dict(in_channels=256, num_classes=NUM_CLASSES),
#             loss_function=torch.nn.functional.cross_entropy,
#             # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
#             optimizer_class=torch.optim.Adam,
#             # Optimizer class class arguments passed to the constructor
#             optimizer_args=dict(lr=2e-4),
#         ),
#     )
# )


CONFIGS = dict(
    sparse_base=SPARSE_BASE,
    static_sparse_weights_only=STATIC_SPARSE_WEIGHTS_ONLY,
    static_sparse_activations_only=STATIC_SPARSE_ACTIVATIONS_ONLY,
    static_sparse_weights_and_activations=STATIC_SPARSE_WEIGHTS_AND_ACTIVATIONS,
    large_static_sparse_weights_and_activations=LARGE_SPARSE_WEIGHTS_ACTIVATIONS,
    static_sparse_first_layer_grid_search=STATIC_SPARSE_FIRST_LAYER_GRID_SEARCH,
    static_sparse_first_layer_full_sparsity=STATIC_SPARSE_FIRST_LAYER_FULL_SPARSITY,
    # large_sparse_grid_search=LARGE_SPARSE_GRID_SEARCH,
)
