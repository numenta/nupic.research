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
"""
debug_finetuning = deepcopy(transformers_base)
debug_finetuning.update(
    # logging
    run_name="debug_finetuning",

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
    run_name="debug_finetuning_bert100k_ntasks",
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
    run_name="finetuning_bert700k_glue",
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
    task_hyperparams=dict(
        mrpc=dict(num_train_epochs=5, num_runs=3),
        wnli=dict(num_train_epochs=5, num_runs=5),
        cola=dict(num_runs=5),
        stsb=dict(num_runs=3),
        rte=dict(num_runs=5),
    ),

)

finetuning_bert100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert100k_glue.update(
    # logging
    run_name="finetuning_bert100k_glue",
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501
)

finetuning_bert1mi_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert1mi_glue.update(
    # logging
    run_name="finetuning_bert1mi_glue",
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
)

finetuning_bert100k_single_task = deepcopy(finetuning_bert100k_glue)
finetuning_bert100k_single_task.update(
    # logging
    task_name=None,
    task_names=["mnli"],
    run_name="finetuning_bert100k_single_task",
)


finetuning_bert700k_single_task = deepcopy(finetuning_bert700k_glue)
finetuning_bert700k_single_task.update(
    # logging
    task_name=None,
    task_names=["mnli", "cola"],
    run_name="finetuning_bert700k_single_task",
)

# Export configurations in this file
CONFIGS = dict(
    debug_finetuning=debug_finetuning,
    debug_finetuning_bert100k=debug_finetuning_bert100k,
    debug_finetuning_bert100k_ntasks=debug_finetuning_bert100k_ntasks,
    finetuning_bert100k_glue=finetuning_bert100k_glue,
    finetuning_bert100k_single_task=finetuning_bert100k_single_task,
    finetuning_bert700k_glue=finetuning_bert700k_glue,
    finetuning_bert700k_single_task=finetuning_bert700k_single_task,
    finetuning_bert1mi_glue=finetuning_bert1mi_glue,
)
