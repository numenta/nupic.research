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

import torch

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from projects.imagenet.experiments_default import DEFAULT

"""
Imagenet superconvergence with compositions of learning schedules
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


DEBUG_COMPOSED_LR = copy.deepcopy(DEFAULT)
DEBUG_COMPOSED_LR.update(
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
                    weight_decay=0.00005,
                    momentum=0.0,
                    nesterov=False,
                ),
            ),
            "30": dict(
                optimizer_args=dict(
                    weight_decay=0.0001,
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


# Two different random seeds and decreasing weight decay,
# as in http://arxiv.org/abs/1711.04291
SPARSE100_WEIGHT_DECAY = copy.deepcopy(DEFAULT)
SPARSE100_WEIGHT_DECAY.update(dict(
    batch_norm_weight_decay=False,
    init_batch_norm=True,
    epochs=60,

    num_classes=100,

    model_args=dict(config=dict(
        num_classes=100,
        defaults_sparse=True,
        activation_params_func=my_auto_sparse_activation_params,
        conv_params_func=my_auto_sparse_conv_params,
        linear_params_func=my_auto_sparse_linear_params
    )),

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
                    weight_decay=0.00005,
                    momentum=0.0,
                    nesterov=False,
                ),
            ),
            "15": dict(
                optimizer_args=dict(
                    weight_decay=0.0001,
                ),
            ),
            "30": dict(
                optimizer_args=dict(
                    weight_decay=0.0002,
                ),
            ),
            "39": dict(
                lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
                lr_scheduler_args=dict(step_size=1, gamma=0.75),
                optimizer_args=dict(
                    lr=0.3,
                    weight_decay=0.0004,
                    momentum=0.0,
                    nesterov=False,
                ),
            ),
        }
    )
))


# Export all configurations
CONFIGS = dict(
    debug_composed_lr=DEBUG_COMPOSED_LR,
    sparse_100_weight_decay=SPARSE100_WEIGHT_DECAY,
)
