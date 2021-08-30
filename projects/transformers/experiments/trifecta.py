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

from callbacks import RezeroWeightsCallback, TrackEvalMetrics
from trainer_mixins import (
    DistillationTrainerMixin,
    LRRangeTestMixin,
    OneCycleLRMixin,
    RigLMixin,
)

from .finetuning import (
    finetuning_bert700k_glue,
    finetuning_bert_100k_glue_get_info,
    finetuning_small_bert_large_dataset_100k_glue,
)
from .sparse_bert import fully_static_sparse_bert_100k_fp16
from .sparse_bertitos import small_bert_sparse_100k, tiny_bert_sparse_100k


"""
These are configs that combine three strong approaches to sparse training
    1. Knowledge Distillation
    2. RigL Dynamic Sparsity
    3. One Cycle Learning Rate
"""


class KDRigLOneCycleLRTrainer(OneCycleLRMixin,
                              DistillationTrainerMixin,
                              RigLMixin,
                              Trainer):
    pass


class KDLRRangeTestTrainer(LRRangeTestMixin,
                           DistillationTrainerMixin,
                           Trainer):
    pass


lr_range_test_args = dict(
    max_steps=100,
    trainer_class=KDLRRangeTestTrainer,

    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.05,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
    overwrite_output_dir=True,
    do_eval=True,
)

# ---------
# Tiny BERT
# ---------

# This combines KD + RigL + OneCycle LR on Tiny BERT.
tiny_bert_trifecta_300k = deepcopy(tiny_bert_sparse_100k)
tiny_bert_trifecta_300k.update(
    max_steps=300000,
    model_type="fully_static_sparse_bert",
    overwrite_output_dir=True,

    # Sparsity callback
    trainer_callbacks=[
        RezeroWeightsCallback(),
        TrackEvalMetrics(),
    ],
    fp16=True,

    trainer_class=KDRigLOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.0075,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],

        # RigL
        prune_fraction=0.3,
        prune_freq=100,
    ),
)

tiny_bert_trifecta_100k = deepcopy(tiny_bert_trifecta_300k)
tiny_bert_trifecta_100k.update(
    max_steps=100000,
)


# This fine-tunes a pretrained model from `tiny_bert_trifecta_100k`
finetuning_tiny_bert_trifecta_100k = deepcopy(finetuning_bert700k_glue)
finetuning_tiny_bert_trifecta_100k.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/tiny_bert_trifecta_100k",  # noqa: E501
    trainer_callbacks=[
        RezeroWeightsCallback(),
        TrackEvalMetrics(),
    ],
)


# LR Range Test for training with KD and OneCycle LR. It's assumed the observed max_lr
# will carry over to training with RigL.
tiny_bert_trifecta_lr_range_test = deepcopy(tiny_bert_trifecta_300k)
tiny_bert_trifecta_lr_range_test.update(
    max_steps=100,
    trainer_class=KDLRRangeTestTrainer,
    # eval_steps=1,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.05,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
    overwrite_output_dir=True,
    do_eval=True,
)

# ---------
# Small BERT
# ---------


# Dataset used for BERT-Base but with max_seq_length made smaller.
small_bert_dataset_args = dict(
    max_seq_length=128,
    dataset_name=("wikipedia", "bookcorpus"),
    dataset_config_name=("20200501.en", None),
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",  # noqa: E501
)


# This combines KD + RigL + OneCycle LR on Small BERT.
# This gets a NaN eval-loss for max_lr=0.006
small_bert_trifecta_300k = deepcopy(small_bert_sparse_100k)
small_bert_trifecta_300k.update(
    max_steps=300000,
    model_type="fully_static_sparse_bert",
    overwrite_output_dir=True,

    # Using batch_size of 16 instead of 128 since we're training on 8 GPUs.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    # Sparsity callback
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,

    trainer_class=KDRigLOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.003,
        pct_start=0.10,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],

        # RigL
        prune_fraction=0.3,
        prune_freq=100,
    ),
)


# Small BERT 80% Sparse Trifecta Model
small_bert_trifecta_100k = deepcopy(small_bert_trifecta_300k)
small_bert_trifecta_100k.update(
    max_steps=100000,
    **small_bert_dataset_args,
)
small_bert_trifecta_100k["trainer_mixin_args"].update(
    # The LR Range test suggests to use max_lr=0.0076
    # But using a small one as `small_bert_trifecta_300k`
    # resulted in NaN loss with a similar sparsity and lr.
    max_lr=0.005,
)
small_bert_trifecta_100k["config_kwargs"].update(
    # This will have 5,481,940 params, actual sparsity will be 80.08%
    sparsity=0.8027,
)


# LR Range test for `small_bert_trifecta_100k`
# Results here: https://wandb.ai/numenta/huggingface/runs/3ocz9yac
small_bert_trifecta_lr_range_test = deepcopy(small_bert_trifecta_100k)
small_bert_trifecta_lr_range_test.update(
    **lr_range_test_args
)


# Small BERT 85% Sparse Trifecta Model
small_bert_trifecta_85_100k = deepcopy(small_bert_trifecta_100k)
small_bert_trifecta_85_100k["trainer_mixin_args"].update(
    # The LR Range test suggests to use max_lr=0.0086
    # As with `small_bert_trifecta_100k`, we'll use a slightly
    # smaller lr than the test suggests.
    max_lr=0.006,
)
small_bert_trifecta_85_100k.update(
    max_steps=100000,
)
small_bert_trifecta_85_100k["config_kwargs"].update(
    # This will have 4,128,460 params, actual sparsity will be 85.13%
    sparsity=0.8529,
)


# LR Range test for `small_bert_trifecta_85_100k`
# Results here: https://wandb.ai/numenta/huggingface/runs/1pfur4bb
small_bert_trifecta_85_lr_range_test = deepcopy(small_bert_trifecta_85_100k)
small_bert_trifecta_85_lr_range_test.update(
    **lr_range_test_args
)


# Small BERT 90% Sparse Trifecta Model
small_bert_trifecta_90_100k = deepcopy(small_bert_trifecta_100k)
small_bert_trifecta_90_100k["trainer_mixin_args"].update(
    # The LR Range test suggests to use max_lr=0.01
    # As with `small_bert_trifecta_100k`, we'll use a slightly
    # smaller lr than the test suggests.
    max_lr=0.007,
)
small_bert_trifecta_90_100k.update(
    max_steps=100000,
)
small_bert_trifecta_90_100k["config_kwargs"].update(
    # This will have 2,745,672 params, actual sparsity will be 90.02%
    sparsity=0.90309,
)


# LR Range test for `small_bert_trifecta_90_100k`
# Results here: https://wandb.ai/numenta/huggingface/runs/1m9bcglt
small_bert_trifecta_90_lr_range_test = deepcopy(small_bert_trifecta_90_100k)
small_bert_trifecta_90_lr_range_test.update(
    **lr_range_test_args
)


# Small BERT 2x Wide Sparse Trifecta Model
small_bert_trifecta_2x_100k = deepcopy(small_bert_trifecta_100k)
small_bert_trifecta_2x_100k["trainer_mixin_args"].update(
    # The LR Range test suggests to use max_lr=0.01
    # As with `small_bert_trifecta_100k`, we'll use a slightly
    # smaller lr than the test suggests.
    max_lr=0.006,
)
small_bert_trifecta_2x_100k.update(
    max_steps=100000,
)
small_bert_trifecta_2x_100k["config_kwargs"].update(
    # This will have 4,057,928 params, actual sparsity will be 94.94%
    # Note that the 85% model has 4,128,460 which this tries to mimic.
    hidden_size=512 * 2,
    intermediate_size=2048 * 2,
    sparsity=0.9507,
)


# LR Range test for `small_bert_trifecta_90_100k`
# Results here: https://wandb.ai/numenta/huggingface/runs/18luah0e
small_bert_trifecta_2x_lr_range_test = deepcopy(small_bert_trifecta_2x_100k)
small_bert_trifecta_2x_lr_range_test.update(
    **lr_range_test_args
)


# Small BERT 4x Wide Sparse Trifecta Model
small_bert_trifecta_4x_100k = deepcopy(small_bert_trifecta_100k)
small_bert_trifecta_4x_100k["trainer_mixin_args"].update(
    # The LR Range test suggests to use max_lr=0.013
    # As with `small_bert_trifecta_100k`, we'll use a slightly
    # smaller lr than the test suggests.
    max_lr=0.008,
)
small_bert_trifecta_4x_100k.update(
    max_steps=100000,
)
small_bert_trifecta_4x_100k["config_kwargs"].update(
    # This will have 2,686,988 params, actual sparsity will be 98.97%
    # Note that the 90% model has 2,745,672 which this tries to mimic.
    hidden_size=512 * 4,
    intermediate_size=2048 * 4,
    sparsity=0.9909,
)


# LR Range test for `small_bert_trifecta_90_100k`
# Results here: https://wandb.ai/numenta/huggingface/runs/2kwe9dic
small_bert_trifecta_4x_lr_range_test = deepcopy(small_bert_trifecta_4x_100k)
small_bert_trifecta_4x_lr_range_test.update(
    **lr_range_test_args
)

# ---------
# Small BERT finetuning
# ---------

finetuning_small_bert_trifecta_100k_glue = deepcopy(
    finetuning_small_bert_large_dataset_100k_glue)
finetuning_small_bert_trifecta_100k_glue.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_80%_trifecta_100k",  # noqa: E501
    trainer_callbacks=[
        TrackEvalMetrics(),
        RezeroWeightsCallback()],
    rm_checkpoints=True,
)
finetuning_small_bert_trifecta_100k_glue["task_hyperparams"]["mnli"].update(
    trainer_callbacks=[TrackEvalMetrics(n_eval_sets=2),
                       RezeroWeightsCallback()],
)

finetuning_small_bert_sparse_85_trifecta_100k_glue = deepcopy(
    finetuning_small_bert_trifecta_100k_glue)
finetuning_small_bert_sparse_85_trifecta_100k_glue.update(
    # Model arguments
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_85%_trifecta_100k",  # noqa: E501
)


finetuning_small_bert_sparse_90_trifecta_100k_glue = deepcopy(
    finetuning_small_bert_trifecta_100k_glue)
finetuning_small_bert_sparse_90_trifecta_100k_glue.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_90%_trifecta_100k"  # noqa: E501
)


finetuning_small_bert_sparse_2x_trifecta_100k_glue = deepcopy(
    finetuning_small_bert_trifecta_100k_glue)
finetuning_small_bert_sparse_2x_trifecta_100k_glue.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_2x_trifecta_100k"  # noqa: E501
)


finetuning_small_bert_sparse_4x_trifecta_100k_glue = deepcopy(
    finetuning_small_bert_trifecta_100k_glue)
finetuning_small_bert_sparse_4x_trifecta_100k_glue.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_4x_trifecta_100k"  # noqa: E501
)

# ---------
# BERT Base
# ---------


# BERT Base with KD + RigL + OneCycle LR
# This achieves and eval-loss of 2.138, just slightly under 2.154 from its dense
# counterpart. See `sparse_v5_trifecta_100k` in the README for more details.
# This took 18h 21m on four p3dn.24xlarges.
bert_sparse_trifecta_100k = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_trifecta_100k.update(
    trainer_class=KDRigLOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.0012,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],

        # RigL
        prune_fraction=0.3,
        prune_freq=100,
    ),
    overwrite_output_dir=True,
)


# The is like the one above, but 85% sparse
bert_sparse_85_trifecta_100k = deepcopy(bert_sparse_trifecta_100k)
bert_sparse_85_trifecta_100k["config_kwargs"].update(
    sparsity=0.85,
)
bert_sparse_85_trifecta_100k["trainer_mixin_args"].update(
    max_lr=0.001,
)


# The is like the one above, but 90% sparse
bert_sparse_90_trifecta_100k = deepcopy(bert_sparse_trifecta_100k)
bert_sparse_90_trifecta_100k["config_kwargs"].update(
    sparsity=0.90,
)
bert_sparse_90_trifecta_100k["trainer_mixin_args"].update(
    max_lr=0.0012,
)


verify_bert_sparse_trifecta_100k = deepcopy(bert_sparse_trifecta_100k)
verify_bert_sparse_trifecta_100k.update(
    # Training arguments
    do_train=False,
    do_eval=True,
    overwrite_output_dir=False,
    save_tokenized_data=False,
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",  # noqa: E501
)

# ---------
# BERT Base finetuning
# ---------


# This fine-tunes a pretrained model from `bert_sparse_trifecta_100k` above.
finetuning_bert_sparse_trifecta_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert_sparse_trifecta_100k_glue.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_trifecta_100k",  # noqa: E501
    trainer_callbacks=[
        RezeroWeightsCallback(),
        TrackEvalMetrics(),
        ],
    rm_checkpoints=True,
)


# 80% sparse, warmup, long runs
finetuning_bert_sparse_trifecta_100k_glue_get_info = deepcopy(
    finetuning_bert_100k_glue_get_info)
finetuning_bert_sparse_trifecta_100k_glue_get_info.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_trifecta_100k",  # noqa: E501
    trainer_callbacks=[
        RezeroWeightsCallback(),
        TrackEvalMetrics()],
    warmup_ratio=0.1,
    rm_checkpoints=True,
)
finetuning_bert_sparse_trifecta_100k_glue_get_info["task_hyperparams"]["mnli"].update(
    trainer_callbacks=[TrackEvalMetrics(n_eval_sets=2),
                       RezeroWeightsCallback()],
)

# As above, but 85% sparse
finetuning_bert_sparse_85_trifecta_100k_glue_get_info = deepcopy(
    finetuning_bert_sparse_trifecta_100k_glue_get_info)
finetuning_bert_sparse_85_trifecta_100k_glue_get_info.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_85%_trifecta_100k"  # noqa: E501
)

# As above, but 90% sparse
finetuning_bert_sparse_90_trifecta_100k_glue_get_info = deepcopy(
    finetuning_bert_sparse_trifecta_100k_glue_get_info)
finetuning_bert_sparse_90_trifecta_100k_glue_get_info.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_90%_trifecta_100k"  # noqa: E501
)

# As above, trifecta_2x
finetuning_bert_sparse_trifecta_2x_get_info = deepcopy(
    finetuning_bert_sparse_trifecta_100k_glue_get_info)
finetuning_bert_sparse_trifecta_2x_get_info.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_2x_trifecta_100k",  # noqa: E501
)

# BERT Base 2x Wide
# This is ran with four p4d.24xlarge instances with 8 GPUs each for an effective batch
# size of 256. GPU memory ~26 GB. Time to completion is roughly 35 hrs.
bert_sparse_trifecta_2x_100k = deepcopy(bert_sparse_trifecta_100k)
bert_sparse_trifecta_2x_100k.update(
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    overwrite_output_dir=False,
    save_steps=500,
)
bert_sparse_trifecta_2x_100k["trainer_mixin_args"].update(
    # The lr range test suggested to use 0.0023, but that resulted in
    # unstable training, so this lower lr will be used.
    max_lr=0.0015,
)
bert_sparse_trifecta_2x_100k["config_kwargs"].update(
    hidden_size=768 * 2,
    intermediate_size=3072 * 2,
    sparsity=0.9595,  # this will have 16.56 M on-params
)


# LR range test for bert_sparse_trifecta_2x_100k
# Run: https://wandb.ai/numenta/huggingface/runs/e7iurznf
bert_sparse_2x_100k_kd_lr_range_test = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_2x_100k_kd_lr_range_test.update(
    max_steps=100,
    per_device_train_batch_size=8,
    per_evice_eval_batch_size=8,

    trainer_class=KDLRRangeTestTrainer,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.005,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)
bert_sparse_2x_100k_kd_lr_range_test["config_kwargs"].update(
    hidden_size=768 * 2,
    intermediate_size=3072 * 2,
    sparsity=0.9595,  # this will have 16.56 M on-params
)


# BERT Base 4x Wide
# This config is still in progress. We've yet to understand the memory requirements
# and whether some number of p4d.24xlarge can be used to complete the run.
bert_sparse_trifecta_4x_100k = deepcopy(bert_sparse_trifecta_100k)
bert_sparse_trifecta_4x_100k.update(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",

    # We'll skip eval in this config as it takes a lot more GPU memory.
    do_eval=False,
)
bert_sparse_trifecta_4x_100k["trainer_mixin_args"].update(
    max_lr=0.005,


    # RigL
    prune_fraction=0.3,
    prune_freq=100,
)
bert_sparse_trifecta_4x_100k["config_kwargs"].update(
    hidden_size=768 * 4,
    intermediate_size=3072 * 4,
    sparsity=0.99365,  # this will have 11.41 M on-params
)


# LR range test for bert_sparse_trifecta_4x_100k
# Run: https://wandb.ai/numenta/huggingface/runs/uipplb2n
# This is ran on four p4d.24xlarge instances for an effective batch size of 256
bert_sparse_4x_100k_kd_lr_range_test = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_4x_100k_kd_lr_range_test.update(
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,

    trainer_class=KDLRRangeTestTrainer,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.01,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)
bert_sparse_4x_100k_kd_lr_range_test["config_kwargs"].update(
    hidden_size=768 * 4,
    intermediate_size=3072 * 4,
    sparsity=0.99365,  # this will have 11 M on-params
)


CONFIGS = dict(
    # Tiny BERT
    tiny_bert_trifecta_100k=tiny_bert_trifecta_100k,
    tiny_bert_trifecta_300k=tiny_bert_trifecta_300k,
    tiny_bert_trifecta_lr_range_test=tiny_bert_trifecta_lr_range_test,
    finetuning_tiny_bert_trifecta_100k=finetuning_tiny_bert_trifecta_100k,

    #   80% sparse
    small_bert_trifecta_100k=small_bert_trifecta_100k,
    small_bert_trifecta_300k=small_bert_trifecta_300k,
    small_bert_trifecta_lr_range_test=small_bert_trifecta_lr_range_test,
    #   85% sparse
    small_bert_trifecta_85_100k=small_bert_trifecta_85_100k,
    small_bert_trifecta_85_lr_range_test=small_bert_trifecta_85_lr_range_test,
    finetuning_small_bert_sparse_85_trifecta_100k_glue=finetuning_small_bert_sparse_85_trifecta_100k_glue,  # noqa: E501
    #   90% sparse
    small_bert_trifecta_90_100k=small_bert_trifecta_90_100k,
    small_bert_trifecta_90_lr_range_test=small_bert_trifecta_90_lr_range_test,
    finetuning_small_bert_sparse_90_trifecta_100k_glue=finetuning_small_bert_sparse_90_trifecta_100k_glue,  # noqa: E501
    #   2x wide
    small_bert_trifecta_2x_100k=small_bert_trifecta_2x_100k,
    small_bert_trifecta_2x_lr_range_test=small_bert_trifecta_2x_lr_range_test,
    finetuning_small_bert_sparse_2x_trifecta_100k_glue=finetuning_small_bert_sparse_2x_trifecta_100k_glue,  # noqa: E501
    #   4x wide
    small_bert_trifecta_4x_100k=small_bert_trifecta_4x_100k,
    small_bert_trifecta_4x_lr_range_test=small_bert_trifecta_4x_lr_range_test,
    finetuning_small_bert_sparse_4x_trifecta_100k_glue=finetuning_small_bert_sparse_4x_trifecta_100k_glue,  # noqa: E501

    # BERT Base
    #   80% sparse
    bert_sparse_trifecta_100k=bert_sparse_trifecta_100k,
    finetuning_bert_sparse_trifecta_100k_glue=finetuning_bert_sparse_trifecta_100k_glue,  # noqa: E501
    finetuning_bert_sparse_trifecta_100k_glue_get_info=finetuning_bert_sparse_trifecta_100k_glue_get_info,  # noqa: E501
    verify_bert_sparse_trifecta_100k=verify_bert_sparse_trifecta_100k,
    #   85% sparse
    bert_sparse_85_trifecta_100k=bert_sparse_85_trifecta_100k,
    finetuning_bert_sparse_85_trifecta_100k_glue_get_info=finetuning_bert_sparse_85_trifecta_100k_glue_get_info,  # noqa: E501
    #   90% sparse
    bert_sparse_90_trifecta_100k=bert_sparse_90_trifecta_100k,
    finetuning_bert_sparse_90_trifecta_100k_glue_get_info=finetuning_bert_sparse_90_trifecta_100k_glue_get_info,  # noqa: E501
    #   2x wide ~16 Mi Params
    bert_sparse_trifecta_2x_100k=bert_sparse_trifecta_2x_100k,
    bert_sparse_2x_100k_kd_lr_range_test=bert_sparse_2x_100k_kd_lr_range_test,
    finetuning_bert_sparse_trifecta_2x_get_info=finetuning_bert_sparse_trifecta_2x_get_info,  # noqa E501
    #   4x wide ~11 Mi Params
    bert_sparse_trifecta_4x_100k=bert_sparse_trifecta_4x_100k,
    bert_sparse_4x_100k_kd_lr_range_test=bert_sparse_4x_100k_kd_lr_range_test,
)
