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

from copy import deepcopy
import pathlib
import pickle
import os

from callbacks import TrackEvalMetrics

from .base import bert_base
# use bert_100k for task-specific hp search prototyping
from .finetuning import finetuning_bert100k_glue
from .trifecta import finetuning_bert_sparse_85_trifecta_100k_glue_get_info

# Get all experiments where hyperparameters have been extracted
exp_dir = pathlib.Path(__file__).parent.resolve()
hp_dir = os.path.join(exp_dir, "hp_finetuning")
hp_files = os.listdir(hp_dir)
HP_SETS = {i: None for i in hp_files if os.path.isdir(os.path.join(hp_dir, i))}

# For each experiment, unpickle task-specific hyperparameters
for model in HP_SETS.keys():
    hp_path = os.path.join(hp_dir, model)
    tasks = os.listdir(hp_path)
    tasks = [i for i in tasks if i.split("_")[-1] == "hps.p"]
    task_hps = {}
    for task in tasks:
        task_name = os.path.basename(task)
        with open(task, "rb") as f:
            task_hps[task_name] = pickle.load(f)
    
    HP_SETS[model] = task_hps

def update_task_hyperparams(local_config, model_name):
    for task in HP_SETS[model_name]:
        for param, value in HP_SETS[model_name][task].items():
            local_config["task_hyperparams"][task][param] = value

    return local_config


# Specify a new config and then use HP_SETS to set task-specific hyperparameters
trifecta_85_hp_chase = deepcopy(finetuning_bert_sparse_85_trifecta_100k_glue_get_info)
trifecta_85_hp_chase = update_task_hyperparams(trifecta_85_hp_chase, 'trifecta_85')

CONFIGS = dict(
    trifecta_85_hp_chase=trifecta_85_hp_chase
)
