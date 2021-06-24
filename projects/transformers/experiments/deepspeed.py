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

from .ablations import tiny_bert_sparse_100k_onecycle_lr_kd
from .distillation import DistillationTrainer

# from .trifecta import tiny_bert_trifecta_100k

# Deepspeed stage 2 default arguments. Args marked with "auto" will be replaced
# with huggingface's values.
DEEPSPEED_STAGE2_ARGS = {
    "deepspeed": {
        # "wall_clock_breakdown": True,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": "True",
            },
            "overlap_comm": True,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "sparse_gradients": True,
        "steps_per_print": 100,
    },
}

# Deepspeed version of tiny_bert_trifecta_100k
# FIXME: This configuration fails because RigL needs to access the gradients
# tiny_bert_trifecta_100k_deepspeed = deepcopy(tiny_bert_trifecta_100k)
# tiny_bert_trifecta_100k_deepspeed.update(DEEPSPEED_STAGE2_ARGS)

# tiny_bert_100k with KD, OneCycleLR and deepspeed
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed = deepcopy(tiny_bert_sparse_100k_onecycle_lr_kd)  # noqa: E501
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed.update(DEEPSPEED_STAGE2_ARGS)

# LR Range Test for tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed
tiny_bert_sparse_100k_kd_lr_range_test_deepspeed = deepcopy(tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed)  # noqa: E501
tiny_bert_sparse_100k_kd_lr_range_test_deepspeed.update(
    # Replace KDOneCycleLRTrainer with just KD trainer for LR Range Test
    trainer_class=DistillationTrainer,
    do_eval=True,
    learning_rate=1,
    max_steps=100,
    logging_steps=1,
    eval_steps=1,
    evaluation_strategy="steps",
)
tiny_bert_sparse_100k_kd_lr_range_test_deepspeed["deepspeed"].update(
    scheduler={
        "type": "LRRangeTest",
        "params": {
            # Pytorch starting LR is computed by dividing min_lr by div_faction
            "lr_range_test_min_lr": 0.0001 / 25.0,
            "lr_range_test_step_size": 1,
            # lr_range_test_step_rate = max_lr / (min_lr / div_factor) / (max_steps / step_size)  # noqa: E501
            "lr_range_test_step_rate": 125,  # For max_lr=0.05, the rate is 125.0
            "lr_range_test_staircase": False,
        },
    }
)

CONFIGS = dict(
    # tiny_bert_trifecta_100k_deepspeed=tiny_bert_trifecta_100k_deepspeed,
    tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed=tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed,  # noqa: E501
    tiny_bert_sparse_100k_kd_lr_range_test_deepspeed=tiny_bert_sparse_100k_kd_lr_range_test_deepspeed,  # noqa: E501
)
