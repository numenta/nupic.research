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
from .finetuning import finetuning_bert100k_glue_get_info
from .trifecta import (
    finetuning_bert_sparse_85_trifecta_100k_glue_get_info,
    finetuning_bert_sparse_90_trifecta_100k_glue_get_info,
    finetuning_bert_sparse_trifecta_100k_glue_get_info,
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


# bet_100k
bert_100k_hp_chase = deepcopy(finetuning_bert100k_glue_get_info)
bert_100k_hp_chase = update_task_hyperparams(bert_100k_hp_chase, "bert_100k")

# 80%
trifecta_80_hp_chase = deepcopy(finetuning_bert_sparse_trifecta_100k_glue_get_info)
trifecta_80_hp_chase = update_task_hyperparams(trifecta_80_hp_chase, "trifecta_80")

# 85%
trifecta_85_hp_chase = deepcopy(finetuning_bert_sparse_85_trifecta_100k_glue_get_info)
trifecta_85_hp_chase = update_task_hyperparams(trifecta_85_hp_chase, "trifecta_85")

# 90%
trifecta_90_hp_chase = deepcopy(finetuning_bert_sparse_90_trifecta_100k_glue_get_info)
trifecta_90_hp_chase = update_task_hyperparams(trifecta_90_hp_chase, "trifecta_90")
debug_trifecta_90_hp_chase_mnli = deepcopy(trifecta_90_hp_chase)
debug_trifecta_90_hp_chase_mnli.update(
    task_name=None,
    task_names=["mnli"],
)


CONFIGS = dict(
    bert_100k_hp_chase=bert_100k_hp_chase,
    trifecta_80_hp_chase=trifecta_80_hp_chase,
    trifecta_85_hp_chase=trifecta_85_hp_chase,
    trifecta_90_hp_chase=trifecta_90_hp_chase,
    debug_trifecta_90_hp_chase_mnli=debug_trifecta_90_hp_chase_mnli,
)
