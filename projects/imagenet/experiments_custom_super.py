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
import inspect

import numpy as np
import ray.tune as tune
import torch

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from projects.imagenet.experiments import DEFAULT

"""
Imagenet superconvergence with more custom sparse params
"""
__all__ = ["CUSTOM_CONFIG"]


def my_auto_sparse_conv_params(in_channels, out_channels, kernel_size):
    """
    Custom weight params.
    :return: a dict to pass to `SparseWeights2d`
    """
    weights_per_channel = kernel_size * kernel_size * in_channels
    if weights_per_channel < 100:
        weights_density = 0.7

    elif weights_per_channel < 200:
        weights_density = 0.5

    elif weights_per_channel < 500:
        weights_density = 0.4

    elif weights_per_channel < 1000:
        weights_density = 0.3

    elif weights_per_channel < 2000:
        weights_density = 0.2

    elif weights_per_channel < 4000:
        weights_density = 0.2

    else:
        weights_density = 0.15

    return dict(
        weight_sparsity=weights_density,
    )


def my_auto_sparse_activation_params(in_channels, out_channels, kernel_size):
    """
    A custom auto sparse params function.
    :return: a dict to pass to `KWinners2d` as params.
    """

    percent_on = 1.0
    if kernel_size != 1:
        if out_channels >= 128:
            percent_on = 0.3

    if percent_on >= 1.0:
        return None
    else:
        return dict(
            percent_on=percent_on,
            boost_strength=1.0,
            boost_strength_factor=0.9,
            local=True,
            k_inference_factor=1.0,
        )


def my_auto_sparse_linear_params(input_size, output_size):
    """
    Custom weight params.
    :return: a dict to pass to `SparseWeights`
    """
    return dict(
        weight_sparsity=0.25,
    )


DEBUG_CUSTOM = copy.deepcopy(DEFAULT)
DEBUG_CUSTOM.update(dict(
    epochs=3,
    log_level="debug",
    num_classes=2,

    model_args=dict(config=dict(
        num_classes=2,
        defaults_sparse=True,
        activation_params_func=my_auto_sparse_activation_params,
        conv_params_func=my_auto_sparse_conv_params,
        linear_params_func=my_auto_sparse_linear_params
    )),

    weight_params=inspect.getsource(my_auto_sparse_conv_params),
    activation_params=inspect.getsource(my_auto_sparse_activation_params),
    linear_params=inspect.getsource(my_auto_sparse_linear_params),

))

DEBUG_COMPOSED_LR = copy.deepcopy(DEBUG_CUSTOM)
DEBUG_COMPOSED_LR.update(
    epochs=60,
    progress=True,
    lr_scheduler_class=ComposedLRScheduler,
    lr_scheduler_args=dict(
        schedulers={
            "0": dict(
                lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
                lr_scheduler_args=dict(
                    max_lr=6.0,
                    div_factor=6,  # initial_lr = 1.0
                    final_div_factor=4000,  # min_lr = 0.00025
                    pct_start=4.0 / 40.0,
                    epochs=40,
                    anneal_strategy="linear",
                    max_momentum=0.01,
                    cycle_momentum=False,
                ),
                optimizer_args=dict(
                    lr=0.1,
                    weight_decay=0.0001,
                    momentum=0.0,
                    nesterov=False,
                ),
            ),
            "40": dict(
                lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
                lr_scheduler_args=dict(step_size=2, gamma=0.5),
                optimizer_args=dict(
                    lr=2.3,
                    weight_decay=0.0001,
                    momentum=0.0,
                    nesterov=False,
                ),
            ),
        }
    )
)

# This gets to about 78.6% after 35 epochs
SPARSE100_SUPER1 = copy.deepcopy(DEFAULT)
SPARSE100_SUPER1.update(dict(
    # No weight decay from batch norm modules
    batch_norm_weight_decay=False,
    init_batch_norm=True,

    epochs=40,
    log_level="debug",
    num_classes=100,

    model_args=dict(config=dict(
        num_classes=100,
        defaults_sparse=True,
        activation_params_func=my_auto_sparse_activation_params,
        conv_params_func=my_auto_sparse_conv_params,
        linear_params_func=my_auto_sparse_linear_params
    )),

    # Use a higher learning rate and no momentum for sparse superconvergence
    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
        max_lr=6.0,
        div_factor=6,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=4.0 / 40.0,
        epochs=40,
        anneal_strategy="linear",
        max_momentum=0.01,
        cycle_momentum=False,
    ),

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.0,
        nesterov=False,
    ),

    weight_params=inspect.getsource(my_auto_sparse_conv_params),
    activation_params=inspect.getsource(my_auto_sparse_activation_params),
    linear_params=inspect.getsource(my_auto_sparse_linear_params),

))

# Try different random seeds of the above
SPARSE100_SUPER_SEEDS = copy.deepcopy(SPARSE100_SUPER1)
SPARSE100_SUPER_SEEDS.update(dict(
    # Seed
    seed=tune.sample_from(lambda spec: np.random.randint(1, 10000)),

    # Number of times to sample from the hyperparameter space. If `grid_search` is
    # provided the grid will be repeated `num_samples` of times.
    num_samples=2,
))

# Try much longer number of epochs (with random seeds) - supposed to do better.
SPARSE100_SUPER_LONG = copy.deepcopy(SPARSE100_SUPER_SEEDS)
SPARSE100_SUPER_LONG.update(dict(
    epochs=80,

    lr_scheduler_args=dict(
        max_lr=6.0,
        div_factor=6,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=5.0 / 80.0,
        epochs=80,
        anneal_strategy="linear",
        max_momentum=0.01,
        cycle_momentum=False,
    ),
))

SPARSE1000_SUPER1 = copy.deepcopy(SPARSE100_SUPER1)
SPARSE1000_SUPER1.update(dict(
    num_classes=1000,

    model_args=dict(config=dict(
        num_classes=1000,
        defaults_sparse=True,
        activation_params_func=my_auto_sparse_activation_params,
        conv_params_func=my_auto_sparse_conv_params,
        linear_params_func=my_auto_sparse_linear_params
    )),

))

# Try much longer number of epochs.
# This is one of the tricks that helps, see https://arxiv.org/abs/1711.04291
SPARSE1000_SUPER_LONG = copy.deepcopy(SPARSE100_SUPER_LONG)
SPARSE1000_SUPER_LONG.update(dict(
    num_classes=1000,

    model_args=dict(config=dict(
        num_classes=1000,
        defaults_sparse=True,
        activation_params_func=my_auto_sparse_activation_params,
        conv_params_func=my_auto_sparse_conv_params,
        linear_params_func=my_auto_sparse_linear_params
    )),


    num_samples=1,
    seed=3497,
))

# Export all configurations
CUSTOM_CONFIG = dict(
    debug_custom=DEBUG_CUSTOM,
    debug_composed_lr=DEBUG_COMPOSED_LR,
    sparse100_super1=SPARSE100_SUPER1,
    sparse100_super_seeds=SPARSE100_SUPER_SEEDS,
    sparse1000_super1=SPARSE1000_SUPER1,
    sparse1000_super_long=SPARSE1000_SUPER_LONG,
)
