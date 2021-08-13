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

from transformers import Trainer

from trainer_mixins import DeepspeedTransformerLayerMixin

from .ablations import tiny_bert_sparse_100k_onecycle_lr_kd
from .distillation import DistillationTrainer
from .gmp_bert import bert_1mi_pretrained_gmp_52k, tiny_bert_gmp_100k

__all__ = ["CONFIGS"]


def _inject_trainer_mixin(config, mixin):
    """
    Injects trainer mixin to the given config
    """
    trainer_class = config["trainer_class"]
    assert issubclass(trainer_class, Trainer)
    config["trainer_class"] = type(
        f"{mixin.__name__}{trainer_class.__name__}", (mixin, trainer_class), {}
    )


# Deepspeed stage 2 default arguments. Args marked with "auto" will be replaced
# with huggingface's values.
DEEPSPEED_STAGE2_ARGS = {
    "tokenized_data_cache_dir": "/mnt/datasets/huggingface/preprocessed-datasets/text",
    "deepspeed": {
        "steps_per_print": 100,
        # "wall_clock_breakdown": True,
        "sparse_gradients": True,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
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
        },
        "fp16": {
            "enabled": "auto",
            "initial_scale_power": 15,
        },
    },
}

# Deepspeed version of tiny_bert_trifecta_100k
# FIXME: This configuration fails because RigL needs to access the gradients
# tiny_bert_trifecta_100k_deepspeed = deepcopy(tiny_bert_trifecta_100k)
# tiny_bert_trifecta_100k_deepspeed.update(DEEPSPEED_STAGE2_ARGS)

# Deepspeed version of tiny_bert_gmp_100k
tiny_bert_gmp_100k_deepspeed = deepcopy(tiny_bert_gmp_100k)
tiny_bert_gmp_100k_deepspeed.update(DEEPSPEED_STAGE2_ARGS)

tiny_bert_gmp_100k_fused_transformer_deepspeed = deepcopy(tiny_bert_gmp_100k_deepspeed)
# Replace HF Transformer Layer with Deepspeed transformetn
_inject_trainer_mixin(
    config=tiny_bert_gmp_100k_fused_transformer_deepspeed,
    mixin=DeepspeedTransformerLayerMixin
)

# tiny_bert_100k with KD, OneCycleLR and deepspeed
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed = deepcopy(tiny_bert_sparse_100k_onecycle_lr_kd)  # noqa: E501
tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed.update(DEEPSPEED_STAGE2_ARGS)

# tiny_bert_100k with KD, OneCycleLR, deepspeed and fused transformer
tiny_bert_sparse_100k_onecycle_lr_kd_fused_transformer_deepspeed = deepcopy(tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed)  # noqa: E501
# Replace HF Transformer Layer with Deepspeed transformetn
_inject_trainer_mixin(
    config=tiny_bert_sparse_100k_onecycle_lr_kd_fused_transformer_deepspeed,
    mixin=DeepspeedTransformerLayerMixin
)

# Deepspeed version of bert_1mi_pretrained_gmp_52k
bert_1mi_pretrained_gmp_52k_deepspeed = deepcopy(bert_1mi_pretrained_gmp_52k)
bert_1mi_pretrained_gmp_52k_deepspeed.update(DEEPSPEED_STAGE2_ARGS)

bert_1mi_pretrained_gmp_52k_fused_transformer_deepspeed = deepcopy(bert_1mi_pretrained_gmp_52k_deepspeed)  # noqa: E501
# Replace HF Transformer Layer with Deepspeed transformetn
_inject_trainer_mixin(
    config=bert_1mi_pretrained_gmp_52k_fused_transformer_deepspeed,
    mixin=DeepspeedTransformerLayerMixin
)

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
    },
    # When using fp16 dynamic loss scale, deepspeed will skip the optimizer
    # and LR scheduler steps whenever the loss value overflows (NaN/Inf).
    # Using deepspeed default values the loss will likely overflow on the
    # first few steps as the dynamic loss scale warms up. When the loss
    # overflows, huggingface will detect the LR scheduler step was skipped
    # and return zero as the current learning rate potentially affecting the
    # results of the LR range test. To avoid loss overflow during the LR
    # range test you could use static loss scale or use a smaller initial
    # scale power.
    # See https://www.deepspeed.ai/docs/config-json/#fp16-training-options
    fp16={
        "enabled": "auto",
        "initial_scale_power": 15,
    },
    steps_per_print=1,
)

CONFIGS = dict(
    # Tiny BERT
    tiny_bert_gmp_100k_deepspeed=tiny_bert_gmp_100k_deepspeed,
    tiny_bert_gmp_100k_fused_transformer_deepspeed=tiny_bert_gmp_100k_fused_transformer_deepspeed,  # noqa
    tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed=tiny_bert_sparse_100k_onecycle_lr_kd_deepspeed,  # noqa: E501
    tiny_bert_sparse_100k_onecycle_lr_kd_fused_transformer_deepspeed=tiny_bert_sparse_100k_onecycle_lr_kd_fused_transformer_deepspeed,  # noqa: E501
    tiny_bert_sparse_100k_kd_lr_range_test_deepspeed=tiny_bert_sparse_100k_kd_lr_range_test_deepspeed,  # noqa: E501

    # BERT Base
    bert_1mi_pretrained_gmp_52k_deepspeed=bert_1mi_pretrained_gmp_52k_deepspeed,  # noqa: E501
    bert_1mi_pretrained_gmp_52k_fused_transformer_deepspeed=bert_1mi_pretrained_gmp_52k_fused_transformer_deepspeed,  # noqa: E501
)
