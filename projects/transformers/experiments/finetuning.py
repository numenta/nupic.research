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

import os
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
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/debug_finetuning"),  # noqa: E501

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
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/debug_finetuning_bert100k"),  # noqa: E501
)

debug_finetuning_bert100k_ntasks = deepcopy(debug_finetuning_bert100k)
debug_finetuning_bert100k_ntasks.update(
    # logging
    run_name="debug_finetuning_bert100k_ntasks",
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/debug_finetuning_bert100k_ntasks"),  # noqa: E501
    report_to="tensorboard",
    task_names=["cola", "stsb"],
    task_hyperparams=dict(
        cola=dict(
            learning_rate=1e-4
        )
    ),
    max_steps=80,
    do_predict=False,
)


finetuning_bert700k_glue = deepcopy(transformers_base)
finetuning_bert700k_glue.update(
    # logging
    run_name="finetuning_bert700k_glue",
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/finetuning_bert700k_glue"),  # noqa: E501
    overwrite_output_dir=True,

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
        mrpc=dict(num_train_epochs=5),
        wlni=dict(num_train_epochs=5)
    ),

)

finetuning_bert100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert100k_glue.update(
    # logging
    task_name="glue",
    run_name="finetuning_bert100k_glue",
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/finetuning_bert100k_glue"),  # noqa: E501
    overwrite_output_dir=True,
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501
)

finetuning_bert100k_single_task = deepcopy(finetuning_bert100k_glue)
finetuning_bert100k_single_task.update(
    # logging
    task_name="qqp",
    run_name="finetuning_bert100k_single_task",
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/finetuning_bert100k_single_task"),  # noqa: E501
)


finetuning_bert700k_single_task = deepcopy(finetuning_bert700k_glue)
finetuning_bert700k_single_task.update(
    # logging
    task_name="qqp",
    run_name="finetuning_bert700k_single_task",
    output_dir=os.path.expanduser("~/nta/results/experiments/transformers/finetuning_bert700k_single_task"),  # noqa: E501
)

# Export configurations in this file
CONFIGS = dict(
    debug_finetuning=debug_finetuning,
    debug_finetuning_bert100k=debug_finetuning_bert100k,
    debug_finetuning_bert100k_ntasks=debug_finetuning_bert100k_ntasks,
    finetuning_bert700k_glue=finetuning_bert700k_glue,
    finetuning_bert100k_glue=finetuning_bert100k_glue,
    finetuning_bert100k_single_task=finetuning_bert100k_single_task,
    finetuning_bert700k_single_task=finetuning_bert700k_single_task,
)
