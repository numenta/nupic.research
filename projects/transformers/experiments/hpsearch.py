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

from ray import tune

from .base import bert_base

# use bert_100k for task-specific hp search prototyping
from .finetuning import finetuning_bert100k_glue


def hp_space(trial):
    return dict(
        learning_rate=tune.loguniform(1e-4, 1e-2)
    )


debug_hp_search = deepcopy(bert_base)
debug_hp_search.update(

    finetuning=False,

    #  Data Training arguments
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",

    # Training Arguments
    logging_steps=50,
    warmup_steps=10,
    max_steps=50,
    overwrite_output_dir=True,
    dataloader_drop_last=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_train=True,
    do_eval=True,
    do_predict=False,

    # hyperparameter search
    hp_space=hp_space,  # required
    hp_num_trials=2,  # required
    hp_validation_dataset_pct=0.05,  # default
    hp_extra_kwargs=dict()  # default

)

# TODO fully specify hyperparam configs
# validation percent
# compute_objective for each task
# for each task
#  num_trials, hp_space, hp_compute_objective, hp_validation_pct
#  resources per trial (kp_search_kwargs)

debug_finetuning_hp_search = deepcopy(finetuning_bert100k_glue)
debug_finetuning_hp_search.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa: E501
    task_name="cola",
    task_names=None,
    num_runs=1,
    max_steps=200,
    save_steps=1,
    warmup_ratio=0.1,
    hp_validation_dataset_pct=1.0,
    report_to="none",
    task_hyperparams=dict(
        cola=dict(
            hp_space=lambda trial: dict(learning_rate=tune.loguniform(1e-5, 1e-2)),
            hp_num_trials=3,
            hp_compute_objective=("maximize", "eval_matthews_correlation")
        )
    ),
)

# Export configurations in this file
CONFIGS = dict(
    debug_hp_search=debug_hp_search,
    debug_finetuning_hp_search=debug_finetuning_hp_search
)
