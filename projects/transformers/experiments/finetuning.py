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
Base Transformers Experiment configuration.
"""

from copy import deepcopy

from callbacks import RezeroWeightsCallback, TrackEvalMetrics

from .base import transformers_base

# uncomment if you need direct access to either variable
# from finetuning_constants import REPORTING_METRICS_PER_TASK, TRAIN_SIZES_PER_TASK


# from ..finetuning_constants import (
#     REPORTING_METRICS_PER_TASK,
#     TRAIN_SIZES_PER_TASK,
# )

"""
Expected for qnli 0.9066 acc, 41 min training time. seed may affect result
Achieved ~0.8896 acc, 11min training time on 4 GPUs.

Effective batch size was 128 with 1/4 the number of steps, so results were expected to
be lower than baseline.

See a summary of the Static Sparse Baseline here:
https://wandb.ai/numenta/huggingface/reports/Static-Sparse-Baselines--Vmlldzo1MTY1MTc
"""


debug_finetuning = deepcopy(transformers_base)
debug_finetuning.update(
    # Data arguments
    task_name=None,
    task_names=["wnli", "rte"],
    max_seq_length=128,

    # Model arguments
    finetuning=True,
    model_name_or_path="bert-base-cased",

    # Training arguments
    do_train=True,
    do_eval=True,
    do_predict=True,
    eval_steps=15,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    max_steps=45,  # made very short for fast debugging
    metric_for_best_model="eval_accuracy",
    num_runs=3,
    rm_checkpoints=True,
    trainer_callbacks=[
        TrackEvalMetrics(),
    ],
)


# Test if checks in code will fix incorrect "metric_for_best_model"
debug_finetuning_bert_sparse_80_trifecta_cola = deepcopy(debug_finetuning)
debug_finetuning_bert_sparse_80_trifecta_cola.update(
    # Data arguments
    task_name=None,
    task_names=["cola"],
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_trifecta_100k",  # noqa: E501
    # Training arguments
    evaluation_strategy="steps",
    eval_steps=50,
    learning_rate=1e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",  # cola does not have this metric
    max_steps=16_000,  # 10x previous
    num_runs=3,
    trainer_callbacks=[
        RezeroWeightsCallback(),
        TrackEvalMetrics(),
    ],
)


debug_finetuning_predict = deepcopy(debug_finetuning)
debug_finetuning_predict.update(
    do_train=False,
    do_eval=False,
    do_predict=True
)


"""
Acc for bert 100k: 0.8641 (in 4 GPUs)
"""
debug_finetuning_bert100k = deepcopy(debug_finetuning)
debug_finetuning_bert100k.update(
    # Data arguments
    overwrite_output_dir=True,

    # Model from checkpoint
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501

    # logging
    run_name="debug_finetuning_bert100k",
)


finetuning_bert700k_glue = deepcopy(transformers_base)
finetuning_bert700k_glue.update(
    # logging
    overwrite_output_dir=True,
    override_finetuning_results=True,

    # Data arguments
    task_name="glue",
    max_seq_length=128,

    # Model arguments
    finetuning=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_700k",  # noqa: E501
    do_train=True,
    do_eval=True,
    do_predict=False,
    eval_steps=50,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    # metric_for_best_model="eval_accuracy",
    num_train_epochs=3,
    num_runs=1,
    # set eval_steps and save_steps proportional to dataset size
    task_hyperparams=dict(
        mrpc=dict(num_train_epochs=5, num_runs=3, save_steps=10),
        wnli=dict(num_train_epochs=5, num_runs=10, save_steps=2),
        cola=dict(num_train_epochs=5,
                  num_runs=10,
                  metric_for_best_model="eval_matthews_correlation"),
        stsb=dict(num_runs=3, metric_for_best_model="eval_pearson"),
        rte=dict(num_runs=10),
        mnli=dict(eval_steps=1_000, save_steps=1_000),
        qnli=dict(eval_steps=300, save_steps=300),
        qqp=dict(eval_steps=1_000, save_steps=1_000)
    ),
    trainer_callbacks=[
        TrackEvalMetrics(),
        ],
)

finetuning_bert100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert100k_glue.update(
    # logging
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501
)


steps_50k = 50_000 // 32

# Get info prefix refers to taking a fresh start after fixing bugs that
# lead to flawed interpretation of previous results. Uses an approach
# similar to the "simple but hard to beat" baseline from
#           https://openreview.net/pdf?id=nzpLWnVAyah
# including warmup, small learning rate, and long training times
finetuning_bert100k_glue_get_info = deepcopy(finetuning_bert100k_glue)
finetuning_bert100k_glue_get_info.update(
    task_hyperparams=dict(

        cola=dict(eval_steps=50,
                  max_steps=steps_50k,
                  metric_for_best_model="eval_matthews_correlation",
                  num_runs=10,
                  ),  # 50k / 8500 ~ 6 epochs

        sst2=dict(eval_steps=100,
                  max_steps=10_000,
                  num_runs=5),  # 67k training size > 50k, default 3 epochs
        mrpc=dict(max_steps=steps_50k,
                  num_runs=10,
                  metric_for_best_model="eval_f1"),
        stsb=dict(max_steps=steps_50k * 2,
                  metric_for_best_model="eval_pearson",
                  num_runs=10),  # 50k / 7000 ~ 8 epochs
        qqp=dict(eval_steps=1_000,
                 max_steps=50_000,
                 num_runs=2,
                 metric_for_best_model="eval_f1"),  # run for a long time
        mnli=dict(eval_steps=1_000,
                  max_steps=50_000,
                  num_runs=2),  # run for a long time
        qnli=dict(eval_steps=500,
                  max_steps=25_000,
                  num_runs=5),  # run for a long time
        rte=dict(max_steps=steps_50k,
                 num_runs=10),  # ~ 20 epochs from paper
        wnli=dict(max_steps=50,
                  num_runs=10)  # run for a short time to avoid overfitting
    ),
    trainer_callbacks=[
        TrackEvalMetrics()],
    warmup_ratio=0.1,
    rm_checkpoints=True,
)

finetuning_bert1mi_glue_get_info = deepcopy(finetuning_bert100k_glue_get_info)
finetuning_bert1mi_glue_get_info.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
)

finetuning_bert1mi_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert1mi_glue.update(
    # logging
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
)

finetuning_bert100k_single_task = deepcopy(finetuning_bert100k_glue)
finetuning_bert100k_single_task.update(
    # logging
    task_name=None,
    task_names=["rte", "wnli", "stsb", "mrpc", "cola"],
)


finetuning_tiny_bert50k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_tiny_bert50k_glue.update(
    model_name_or_path="/home/ec2-user"
                       "/nta/results/experiments/transformers/tiny_bert_50k"
)


finetuning_bert700k_single_task = deepcopy(finetuning_bert700k_glue)
finetuning_bert700k_single_task.update(
    # logging
    task_name=None,
    task_names=["rte", "wnli", "stsb", "mrpc", "cola"],
)

finetuning_bert1mi_single_task = deepcopy(finetuning_bert1mi_glue)
finetuning_bert1mi_single_task.update(
    # logging
    task_name=None,
    task_names=["rte", "wnli", "stsb", "mrpc", "cola"],
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
)


finetuning_sparse_bert_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_sparse_bert_100k_glue.update(
    # Model arguments
    model_type="static_sparse_non_attention_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/static_sparse_non_attention_bert_100k",  # noqa: E501
)


finetuning_sparse_encoder_bert_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_sparse_encoder_bert_100k_glue.update(
    # Model arguments
    model_type="static_sparse_encoder_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/static_sparse_encoder_bert_100k",  # noqa: E501
)


finetuning_fully_sparse_bert_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_fully_sparse_bert_100k_glue.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_100k",  # noqa: E501
)


finetuning_mini_sparse_bert_debug = deepcopy(finetuning_bert700k_glue)
finetuning_mini_sparse_bert_debug.update(
    model_type="static_sparse_encoder_bert",
    model_name_or_path="/home/ec2-user/nta/results/experiments/transformers/mini_sparse_bert_debug",  # noqa: E501
)


# Export configurations in this file
CONFIGS = dict(
    debug_finetuning=debug_finetuning,
    debug_finetuning_bert100k=debug_finetuning_bert100k,
    debug_finetuning_bert_sparse_80_trifecta_cola=debug_finetuning_bert_sparse_80_trifecta_cola,  # noqa: E501
    debug_finetuning_predict=debug_finetuning_predict,
    finetuning_bert100k_glue=finetuning_bert100k_glue,
    finetuning_bert100k_glue_get_info=finetuning_bert100k_glue_get_info,
    finetuning_bert100k_single_task=finetuning_bert100k_single_task,
    finetuning_tiny_bert50k_glue=finetuning_tiny_bert50k_glue,
    finetuning_bert700k_glue=finetuning_bert700k_glue,
    finetuning_bert700k_single_task=finetuning_bert700k_single_task,
    finetuning_bert1mi_glue=finetuning_bert1mi_glue,
    finetuning_bert1mi_glue_get_info=finetuning_bert1mi_glue_get_info,
    finetuning_bert1mi_single_task=finetuning_bert1mi_single_task,
    finetuning_sparse_bert_100k_glue=finetuning_sparse_bert_100k_glue,
    finetuning_sparse_encoder_bert_100k_glue=finetuning_sparse_encoder_bert_100k_glue,
    finetuning_mini_sparse_bert_debug=finetuning_mini_sparse_bert_debug,
    finetuning_fully_sparse_bert_100k_glue=finetuning_fully_sparse_bert_100k_glue,
)
