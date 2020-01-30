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

from .base import DEFAULT


"""
An initial set of good parameters for sparse networks, i.e. Release 1.

The sparse 100 configuration here gets between 80.5 to 82% in top-1 accuracy, after 60
epochs and has about 77.5% average weight sparsity.

The sparse 1000 configuration, with the same weight sparsities, gets about 72.23% in
top-1 accuracy after 120 epochs.

"""


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


# This configuration gets between 80.5 to 82% after 60 epochs
SPARSE100_R1 = copy.deepcopy(DEFAULT)
SPARSE100_R1.update(dict(
    # No weight decay from batch norm modules
    batch_norm_weight_decay=False,
    init_batch_norm=True,

    epochs=60,
    checkpoint_freq=1,
    keep_checkpoints_num=2,
    checkpoint_score_attr="training_iteration",
    checkpoint_at_end=True,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),

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
        pct_start=5.0 / 60.0,
        epochs=60,
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

# Try much longer number of epochs that 100 category version and a lower
# weight decay. (see https://arxiv.org/abs/1711.04291)
# With 120 epochs this gets 72.23% top-1 accuracy. Earlier version got
# about 73% with 200 epochs.
SPARSE1000_R1 = copy.deepcopy(SPARSE100_R1)
SPARSE1000_R1.update(dict(

    num_classes=1000,
    epochs=120,
    model_args=dict(config=dict(
        num_classes=1000,
        defaults_sparse=True,
        activation_params_func=my_auto_sparse_activation_params,
        conv_params_func=my_auto_sparse_conv_params,
        linear_params_func=my_auto_sparse_linear_params
    )),

    lr_scheduler_args=dict(
        max_lr=6.0,
        div_factor=6,  # initial_lr = 1.0
        final_div_factor=4000,  # min_lr = 0.00025
        pct_start=5.0 / 120.0,
        epochs=120,
        anneal_strategy="linear",
        max_momentum=0.01,
        cycle_momentum=False,
    ),

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.00005,
        momentum=0.0,
        nesterov=False,
    ),
))

CONFIGS = dict(
    sparse_100_r1=SPARSE100_R1,
    sparse_1000_r1=SPARSE1000_R1,
)
