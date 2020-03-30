#  Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc.  No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.
import copy
import functools
import time

import numpy as np
import ray.tune as tune
import torch

from nupic.hardware.frameworks.pytorch.modules import StructuredSparseWeights
from nupic.hardware.frameworks.pytorch.sparse_layer_params import (
    auto_structured_sparse_conv_params,
)
from nupic.hardware.frameworks.xilinx.key_parameters.r2_params import (
    auto_sparse_activation_params_r2,
    auto_sparse_linear_params_r2,
    auto_sparse_weights_params_r2,
)

from .base import DEFAULT

"""
Configuration like R2 but where learning rates are tuned by SigOpt.
"""


# This gets up to 85.24%% accuracies, our record to date. Checkpoint saved to:
#   /mnt/efs/results/sahmad/checkpoints/structured_sparse100_r3_e27964a4
# Note: these are not the actual final best parameters. I averaged the top four
# experiments to get the values below.
STRUCTURED_SPARSE100_R3 = copy.deepcopy(DEFAULT)
STRUCTURED_SPARSE100_R3.update(dict(
    batch_norm_weight_decay=False,
    init_batch_norm=True,

    num_samples=3,
    epochs=60,
    checkpoint_freq=3,
    keep_checkpoints_num=2,
    checkpoint_score_attr="training_iteration",
    checkpoint_at_end=True,
    seed=tune.sample_from(lambda spec: np.random.randint(2, 10000)),
    launch_time=time.time(),

    num_classes=100,

    model_args=dict(config=dict(
        num_classes=100,
        defaults_sparse=True,
        conv_params_func=functools.partial(
            auto_structured_sparse_conv_params,
            auto_sparse_weight_fct=auto_sparse_weights_params_r2),
        activation_params_func=auto_sparse_activation_params_r2,
        conv_sparse_weights_type=StructuredSparseWeights,
        linear_params_func=auto_sparse_linear_params_r2,
    )),

    lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
    lr_scheduler_args=dict(
        max_lr=6.91,
        div_factor=7.58,  # initial_lr = 0.91
        final_div_factor=3640,  # min_lr = 0.00025
        pct_start=0.23,
        epochs=60,
        anneal_strategy="linear",
        max_momentum=0.77,
        cycle_momentum=False,
    ),

    use_auto_augment=True,

    optimizer_args=dict(
        lr=0.1,
        weight_decay=0.0001,
        momentum=0.77,
        nesterov=False,
    ),

))

# This version gets __ accuracy after 200 epochs
STRUCTURED_SPARSE1000_R3 = copy.deepcopy(STRUCTURED_SPARSE100_R3)
STRUCTURED_SPARSE1000_R3.update(dict(
    num_samples=1,
    epochs=200,
    num_classes=1000,
    checkpoint_freq=3,
    keep_checkpoints_num=4,

    model_args=dict(config=dict(
        num_classes=1000,
        defaults_sparse=True,
        conv_params_func=functools.partial(
            auto_structured_sparse_conv_params,
            auto_sparse_weight_fct=auto_sparse_weights_params_r2),
        activation_params_func=auto_sparse_activation_params_r2,
        conv_sparse_weights_type=StructuredSparseWeights,
        linear_params_func=auto_sparse_linear_params_r2,
    )),

    lr_scheduler_args=dict(
        max_lr=6.91,
        div_factor=7.58,  # initial_lr = 0.91
        final_div_factor=3640,  # min_lr = 0.00025
        pct_start=30.0 / 200.0,
        epochs=200,
        anneal_strategy="linear",
        max_momentum=0.77,
        cycle_momentum=False,
    ),

    epochs_to_validate=[-1, 0, 1, 2, 20, 40, 60, 80, 100, 120, 140, 160, 180, 197,
                        198, 199, 200],
))

# This experiment restarts training
SPARSE1000_R3_CONTINUE = copy.deepcopy(STRUCTURED_SPARSE1000_R3)
SPARSE1000_R3_CONTINUE.update(dict(

    epochs=83,

    model_args=dict(config=dict(
        num_classes=1000,
        defaults_sparse=True,
        conv_params_func=functools.partial(
            auto_structured_sparse_conv_params,
            auto_sparse_weight_fct=auto_sparse_weights_params_r2),
        activation_params_func=auto_sparse_activation_params_r2,
        conv_sparse_weights_type=StructuredSparseWeights,
        linear_params_func=auto_sparse_linear_params_r2,
    )),

    checkpoint_file="/home/.../checkpoint_118/checkpoint",

    lr_scheduler_args=dict(
        max_lr=3.3333,
        div_factor=1.03,  # initial_lr = 3.23
        final_div_factor=12945,  # min_lr = 0.0001. min_lr = initial_lr/final_div_factor
        pct_start=1.0 / 83.0,
        epochs=83,
        anneal_strategy="linear",
        max_momentum=0.77,
        cycle_momentum=False,
    ),

    epochs_to_validate=[-1, 0, 1, 20, 40, 60, 82, 83, 84],

))

# Export all configurations
CONFIGS = dict(
    structured_sparse100_r3=STRUCTURED_SPARSE100_R3,
    structured_sparse1000_r3=STRUCTURED_SPARSE1000_R3,
    sparse1000_r3_continue=SPARSE1000_R3_CONTINUE,
)
