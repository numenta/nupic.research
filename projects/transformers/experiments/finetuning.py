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

from .base import transformers_base

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
    task_name="qnli",
    max_seq_length=128,

    # Model arguments
    finetuning=True,
    model_name_or_path="bert-base-cased",

    # Training arguments
    do_train=True,
    do_eval=True,
    do_predict=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
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

debug_finetuning_bert100k_ntasks = deepcopy(debug_finetuning_bert100k)
debug_finetuning_bert100k_ntasks.update(
    # logging
    report_to="tensorboard",
    task_name="glue",
    # task_name=None,
    # task_names=["cola", "stsb", "mnli"],
    override_finetuning_results=False,
    task_hyperparams=dict(
        wnli=dict(num_runs=2, learning_rate=2e-4),
        rte=dict(num_runs=0),
        cola=dict(num_runs=2),
        stsb=dict(num_runs=1),
    ),
    max_steps=300,
    do_predict=False,
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
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    num_runs=1,
    task_hyperparams=dict(
        mrpc=dict(num_train_epochs=5, num_runs=3),
        wnli=dict(num_train_epochs=5, num_runs=10),
        cola=dict(num_train_epochs=5, num_runs=10),
        stsb=dict(num_runs=3),
        rte=dict(num_runs=10),
    ),
)

finetuning_bert100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert100k_glue.update(
    # logging
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501
)

# the name 'simple' is in reference to the paper
# "On the stability of finetuning BERT"
# where they propose a "simple but hard to beat" approach
# https://openreview.net/pdf?id=nzpLWnVAyah
# num_train_epochs rationale:
# they say rte for 20 is good ~ 50k iterations
# train for 50k iterations unless 1 epoch > 50k samples
# then set to 3 epochs for medium sized like sst2 (67k)
# or set to 1 epoch for large datasets like qqp
finetuning_bert100k_glue_simple = deepcopy(finetuning_bert100k_glue)
finetuning_bert100k_glue_simple.update(
    warmup_ratio=0.1,
    task_hyperparams=dict(
        cola=dict(num_train_epochs=6, num_runs=5),  # 6 * 8500 ~ 50k
        sst2=dict(num_runs=3),  # 67k training size > 50k, default 3 epochs
        mrpc=dict(num_train_epochs=14, num_runs=3),  # 3700 * 14 ~ 51k
        stsb=dict(num_train_epochs=8, num_runs=3),  # 7000*8 > 50k
        # hypothesis for qqp, mnli: training stable < 300k iterations
        # more runs is better than 1 run with more epochs
        qqp=dict(num_train_epochs=1, num_runs=3),  # 300k >> 50k
        mnli=dict(num_train_epochs=1, num_runs=3),  # 300k >> 50k
        qnli=dict(num_runs=3),  # 100k > 50k, defualt to 3 epochs
        rte=dict(num_train_epochs=20, num_runs=3),  # exatly as in paper
        wnli=dict(num_train_epochs=79, num_runs=3)  # large n_epochs to hit > 50k
    )
)

finetuning_bert1mi_glue_simple = deepcopy(finetuning_bert100k_glue_simple)
finetuning_bert1mi_glue_simple.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
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

finetuning_bert1mi_wnli = deepcopy(finetuning_bert100k_single_task)
finetuning_bert1mi_wnli.update(
    task_name=None,
    task_names=["wnli"],
    evaluation_strategy="steps",
    eval_steps=18

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
    debug_finetuning_bert100k_ntasks=debug_finetuning_bert100k_ntasks,
    finetuning_bert100k_glue=finetuning_bert100k_glue,
    finetuning_bert100k_single_task=finetuning_bert100k_single_task,
    finetuning_tiny_bert50k_glue=finetuning_tiny_bert50k_glue,
    finetuning_bert700k_glue=finetuning_bert700k_glue,
    finetuning_bert700k_single_task=finetuning_bert700k_single_task,
    finetuning_bert100k_glue_simple=finetuning_bert100k_glue_simple,
    finetuning_bert1mi_glue=finetuning_bert1mi_glue,
    finetuning_bert1mi_glue_simple=finetuning_bert1mi_glue_simple,
    finetuning_bert1mi_wnli=finetuning_bert1mi_wnli,
    finetuning_bert1mi_single_task=finetuning_bert1mi_single_task,
    finetuning_sparse_bert_100k_glue=finetuning_sparse_bert_100k_glue,
    finetuning_sparse_encoder_bert_100k_glue=finetuning_sparse_encoder_bert_100k_glue,
    finetuning_mini_sparse_bert_debug=finetuning_mini_sparse_bert_debug,
    finetuning_fully_sparse_bert_100k_glue=finetuning_fully_sparse_bert_100k_glue,
)
