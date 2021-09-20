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

import os
import pathlib
from copy import deepcopy

import yaml

# use bert_100k for task-specific hp search prototyping
from .finetuning import finetuning_bert_100k_glue_get_info
from .trifecta import (
    finetuning_bert_sparse_85_trifecta_100k_glue_get_info,
    finetuning_bert_sparse_90_trifecta_100k_glue_get_info,
    finetuning_bert_sparse_trifecta_2x_get_info,
    finetuning_bert_sparse_trifecta_100k_glue_get_info,
    finetuning_small_bert_sparse_2x_trifecta_100k_glue,
    finetuning_small_bert_sparse_4x_trifecta_100k_glue,
    finetuning_small_bert_sparse_85_trifecta_100k_glue,
    finetuning_small_bert_sparse_90_trifecta_100k_glue,
    finetuning_small_bert_trifecta_100k_glue,
)

# Get all experiments where hyperparameters have been extracted
exp_dir = pathlib.Path(__file__).parent.resolve()
hp_dir = os.path.join(exp_dir, "hp_finetuning")
hp_files = os.listdir(hp_dir)
HP_SETS = {i: None for i in hp_files if os.path.isdir(os.path.join(hp_dir, i))}

# For each experiment, unpickle task-specific hyperparameters
for model in HP_SETS.keys():
    hp_path = os.path.join(hp_dir, model)
    tasks = os.listdir(hp_path)
    tasks = [i for i in tasks if i.split("_")[-1] == "hps.yaml"]
    task_hps = {}
    for task in tasks:
        task_name = os.path.basename(task).split("_")[0]
        task_file = os.path.join(hp_path, task)
        with open(task_file, "r") as f:
            task_hps[task_name] = yaml.safe_load(f)

    HP_SETS[model] = task_hps


def update_task_hyperparams(local_config, model_name):
    for task in HP_SETS[model_name]:
        task_name = task.split("_")[0]
        for param, value in HP_SETS[model_name][task_name].items():
            local_config["task_hyperparams"][task_name][param] = value

    return local_config


# ---------
# BERT base variations
# ---------

# bert_100k
bert_100k_hp_chase = deepcopy(finetuning_bert_100k_glue_get_info)
bert_100k_hp_chase = update_task_hyperparams(bert_100k_hp_chase, "bert_100k")

bert_100k_hp_chase_mnli = deepcopy(bert_100k_hp_chase)
bert_100k_hp_chase_mnli.update(
    task_name=None,
    task_names=["mnli"],
)

# 80%
trifecta_80_hp_chase = deepcopy(finetuning_bert_sparse_trifecta_100k_glue_get_info)
trifecta_80_hp_chase = update_task_hyperparams(trifecta_80_hp_chase, "trifecta_80")

trifecta_80_hp_chase_mnli = deepcopy(trifecta_80_hp_chase)
trifecta_80_hp_chase_mnli.update(
    task_name=None,
    task_names=["mnli"],
)

# 85%
trifecta_85_hp_chase = deepcopy(finetuning_bert_sparse_85_trifecta_100k_glue_get_info)
trifecta_85_hp_chase = update_task_hyperparams(trifecta_85_hp_chase, "trifecta_85")

trifecta_85_hp_chase_mnli = deepcopy(trifecta_85_hp_chase)
trifecta_85_hp_chase_mnli.update(
    task_name=None,
    task_names=["mnli"],
)

# 90%
trifecta_90_hp_chase = deepcopy(finetuning_bert_sparse_90_trifecta_100k_glue_get_info)
trifecta_90_hp_chase = update_task_hyperparams(trifecta_90_hp_chase, "trifecta_90")

trifecta_90_hp_chase_mnli = deepcopy(trifecta_90_hp_chase)
trifecta_90_hp_chase_mnli.update(
    task_name=None,
    task_names=["mnli"],
)

trifecta_90_hp_chase_follow_up = deepcopy(trifecta_90_hp_chase)

# 2X
# Original batch size per device was 32, reducing here to 8 since it needs to run on
# a p3.8x, otherwise it will run out of memory.
trifecta_2x_hp_guess = deepcopy(finetuning_bert_sparse_trifecta_2x_get_info)
trifecta_2x_hp_guess = update_task_hyperparams(
    trifecta_2x_hp_guess, "trifecta_2x_guess")
trifecta_2x_hp_guess.update(
    per_device_train_batch_size=32 // 4,
    per_device_eval_batch_size=32 // 4,
)
trifecta_2x_hp_guess_follow_up = deepcopy(trifecta_2x_hp_guess)
trifecta_2x_hp_guess_follow_up.update(
    task_name=None,
    task_names=["mrpc", "cola", "mnli", "qnli", "qqp", "sst2"]
)

# ---------
# BERT small variations
# ---------

small_bert_big_dataset_hp_chase = deepcopy(finetuning_bert_100k_glue_get_info)
small_bert_big_dataset_hp_chase.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_large_dataset_100k",  # noqa: E501
)
small_bert_big_dataset_hp_chase = update_task_hyperparams(
    small_bert_big_dataset_hp_chase, "small_100k")

# 80%
trifecta_80_small_hp_chase = deepcopy(finetuning_small_bert_trifecta_100k_glue)
trifecta_80_small_hp_chase = update_task_hyperparams(trifecta_80_small_hp_chase, "trifecta_small_80")  # noqa: E501

# Temporarily updated to just finetune the remaining tasks, since a typo
# caused these runs to break starting at mrpc
trifecta_80_small_hp_chase.update(
    task_name=None,
    task_names=["mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
)
trifecta_80_small_hp_chase_first_two = deepcopy(trifecta_80_small_hp_chase)
trifecta_80_small_hp_chase_first_two.update(
    task_names=["cola", "mnli"]
)

# 85%
trifecta_85_small_hp_chase = deepcopy(finetuning_small_bert_sparse_85_trifecta_100k_glue)  # noqa: E501
trifecta_85_small_hp_chase = update_task_hyperparams(trifecta_85_small_hp_chase, "trifecta_small_85")  # noqa: E501
trifecta_85_small_hp_chase.update(
    task_name=None,
    task_names=["mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
)

trifecta_85_small_hp_chase_first_two = deepcopy(trifecta_85_small_hp_chase)
trifecta_85_small_hp_chase_first_two.update(
    task_names=["cola", "mnli"]
)

# 90%
trifecta_90_small_hp_chase = deepcopy(finetuning_small_bert_sparse_90_trifecta_100k_glue)  # noqa: E501
trifecta_90_small_hp_chase = update_task_hyperparams(trifecta_90_small_hp_chase, "trifecta_small_90")  # noqa: E501
trifecta_90_small_hp_chase.update(
    task_name=None,
    task_names=["mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
)

trifecta_90_small_hp_chase_first_two = deepcopy(trifecta_90_small_hp_chase)
trifecta_90_small_hp_chase_first_two.update(
    task_names=["cola", "mnli"]
)

trifecta_90_small_hp_chase_debug_first_two = deepcopy(trifecta_90_small_hp_chase_first_two)  # noqa: E501
trifecta_90_small_hp_chase_debug_first_two["task_hyperparams"]["cola"].update(
    max_steps=100,
    num_runs=3,
)
trifecta_90_small_hp_chase_debug_first_two["task_hyperparams"]["mnli"].update(
    max_steps=100,
    num_runs=3,
)

# 2x
trifecta_2x_small_hp_chase = deepcopy(finetuning_small_bert_sparse_2x_trifecta_100k_glue)  # noqa: E501
trifecta_2x_small_hp_chase = update_task_hyperparams(trifecta_2x_small_hp_chase, "trifecta_small_2x")  # noqa: E501
trifecta_2x_small_hp_chase.update(
    task_name=None,
    task_names=["mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
)

trifecta_2x_small_hp_chase_follow_up = deepcopy(trifecta_2x_small_hp_chase)
trifecta_2x_small_hp_chase_follow_up.update(
    task_names=["cola", "mnli", "sst2"]
)
trifecta_2x_small_hp_chase_follow_up["task_hyperparams"]["sst2"].update(
    num_runs=2
)

# 4x
trifecta_4x_small_hp_chase = deepcopy(finetuning_small_bert_sparse_4x_trifecta_100k_glue)  # noqa: E501
trifecta_4x_small_hp_chase = update_task_hyperparams(trifecta_2x_small_hp_chase, "trifecta_small_4x")  # noqa: E501
trifecta_4x_small_hp_chase.update(
    task_name=None,
    task_names=["mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
)

trifecta_4x_small_hp_chase_first_two = deepcopy(trifecta_4x_small_hp_chase)
trifecta_4x_small_hp_chase_first_two.update(
    task_names=["cola", "mnli"]
)

CONFIGS = dict(
    # BERT base
    bert_100k_hp_chase=bert_100k_hp_chase,
    bert_100k_hp_chase_mnli=bert_100k_hp_chase_mnli,
    trifecta_80_hp_chase=trifecta_80_hp_chase,
    trifecta_80_hp_chase_mnli=trifecta_80_hp_chase_mnli,
    trifecta_85_hp_chase=trifecta_85_hp_chase,
    trifecta_85_hp_chase_mnli=trifecta_85_hp_chase_mnli,
    trifecta_90_hp_chase=trifecta_90_hp_chase,
    trifecta_90_hp_chase_mnli=trifecta_90_hp_chase_mnli,
    trifecta_90_hp_chase_follow_up=trifecta_90_hp_chase_follow_up,
    trifecta_2x_hp_guess=trifecta_2x_hp_guess,
    trifecta_2x_hp_guess_follow_up=trifecta_2x_hp_guess_follow_up,

    # BERT small
    small_bert_big_dataset_hp_chase=small_bert_big_dataset_hp_chase,
    trifecta_80_small_hp_chase=trifecta_80_small_hp_chase,
    trifecta_85_small_hp_chase=trifecta_85_small_hp_chase,
    trifecta_90_small_hp_chase=trifecta_90_small_hp_chase,
    trifecta_2x_small_hp_chase=trifecta_2x_small_hp_chase,
    trifecta_4x_small_hp_chase=trifecta_4x_small_hp_chase,

    # BERT small follow ups
    trifecta_80_small_hp_chase_first_two=trifecta_80_small_hp_chase_first_two,
    trifecta_85_small_hp_chase_first_two=trifecta_85_small_hp_chase_first_two,
    trifecta_90_small_hp_chase_first_two=trifecta_90_small_hp_chase_first_two,
    trifecta_2x_small_hp_chase_follow_up=trifecta_2x_small_hp_chase_follow_up,
    trifecta_4x_small_hp_chase_first_two=trifecta_4x_small_hp_chase_first_two,

    # BERT small follow up debugging
    trifecta_90_small_hp_chase_debug_first_two=trifecta_90_small_hp_chase_debug_first_two,  # noqa: E501
)
