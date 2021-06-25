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

finetuning_hp_search = deepcopy(finetuning_bert100k_glue)
finetuning_hp_search.update(

    task_hyperparams=dict(

        cola=dict(
                  hp_space = lambda trial: dict(
                    eval_steps=tune.random
                    learning_rate=tune.
                  )
                  eval_steps=50,
                  max_steps=steps_50k,
                  metric_for_best_model="eval_matthews_correlation",
                  num_runs=5,
                  ),  # 50k / 8500 ~ 6 epochs

        sst2=dict(num_runs=3),  # 67k training size > 50k, default 3 epochs
        mrpc=dict(max_steps=steps_50k, num_runs=3),  # 50k / 3700 ~ 14 epochs

        stsb=dict(max_steps=steps_50k,
                  metric_for_best_model="eval_pearson",
                  num_runs=3),  # 50k / 7000 ~ 8 epochs

        qqp=dict(eval_steps=1_000, num_runs=3),  # 300k >> 50k
        mnli=dict(eval_steps=1_000,
                  num_runs=3),  # 300k >> 50k
        qnli=dict(eval_steps=500,
                  num_runs=3),  # 100k > 50k, defualt to 3 epochs
        rte=dict(max_steps=steps_50k, num_runs=3),  # ~ 20 epochs from paper
        wnli=dict(max_steps=steps_50k, num_runs=3)  # 50k / 634 ~ 79 epochs
    ),
    trainer_callbacks=[
        TrackEvalMetrics()],
    warmup_ratio=0.1,
)

# Export configurations in this file
CONFIGS = dict(
    debug_hp_search=debug_hp_search
)
