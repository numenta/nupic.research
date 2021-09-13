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
from transformers import Trainer

from callbacks import RezeroWeightsCallback, TrackEvalMetrics
from trainer_mixins import MultiEvalSetsTrainerMixin

from .base import bert_base
# use bert_100k for task-specific hp search prototyping
from .finetuning import finetuning_bert100k_glue
from .trifecta import finetuning_bert_sparse_85_trifecta_100k_glue_get_info


def hp_space(trial):
    return dict(
        learning_rate=tune.loguniform(1e-4, 1e-2)
    )


class MultiEvalSetTrainer(MultiEvalSetsTrainerMixin, Trainer):
    pass


# ---------
# Debugging
# ---------

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
    task_name=None,
    task_names=["cola", "rte"],
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
        ),
        rte=dict(
            hp_space=lambda trial: dict(learning_rate=tune.loguniform(1e-5, 1e-2)),
            hp_num_trials=3,
            hp_compute_objective=("maximize", "eval_accuracy")
        ),
    ),
)

debug_finetuning_sparse_hp_search = deepcopy(
    finetuning_bert_sparse_85_trifecta_100k_glue_get_info)
debug_finetuning_sparse_hp_search.update(
    task_name="cola",
    task_names=None,
    num_runs=1,
    max_steps=200,
    save_steps=1,
    warmup_ratio=0.1,
    hp_validation_dataset_pct=1.0,
    report_to="wandb",
    task_hyperparams=dict(
        cola=dict(
            hp_space=lambda trial: dict(
                learning_rate=tune.loguniform(1e-5, 1e-2),
                max_steps=tune.randint(10, 1000),
                warmup_ratio=tune.choice([0., 0.05, 0.1, 0.2, 0.4])),
            hp_num_trials=3,
            hp_compute_objective=("maximize", "eval_matthews_correlation")
        )
    ),
)

debug_finetuning_sparse_hp_mnli = deepcopy(debug_finetuning_sparse_hp_search)
debug_finetuning_sparse_hp_mnli.update(
    task_name="mnli",
    metric_for_best_model="eval_mm_accuracy",
    trainer_class=MultiEvalSetTrainer,
    trainer_mixin_args=dict(
        eval_sets=["validation_matched", "validation_mismatched"],
        eval_prefixes=["eval", "eval_mm"],
    ),
    hp_num_trials=2,
    trainer_callbacks=[TrackEvalMetrics(n_eval_sets=2),
                       RezeroWeightsCallback()],
    task_hyperparams=dict(
        mnli=dict(
            hp_space=lambda trial: dict(
                learning_rate=tune.loguniform(1e-6, 1e-3),
                # max_steps = tune.grid_search([100, 200]),
            ),
            hp_num_trials=2,
            hp_compute_objective=("maximize", "eval_mm_accuracy"),
            eval_steps=15,
            max_steps=45,
        )
    )
)

# ---------
# BERT small, small tasks
# ---------

hp_search_finetuning_small_bert_100k_small_tasks = deepcopy(
    debug_finetuning_hp_search)
hp_search_finetuning_small_bert_100k_small_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_large_dataset_100k",  # noqa: E501
    task_name=None,
    task_names=["cola", "mrpc", "rte", "stsb", "wnli"],
    hp_space=lambda trial: dict(
        learning_rate=tune.loguniform(1e-6, 1e-2),
        max_steps=tune.randint(100, 5000),
        # if warmup 0, no learning rate scheduler
        warmup_ratio=tune.choice([0., 0.1]),
    ),
    hp_num_trials=30,
    task_hyperparams=dict(
        cola=dict(hp_compute_objective=("maximize", "eval_matthews_correlation")),
        mrpc=dict(hp_compute_objective=("maximize", "eval_f1")),
        rte=dict(hp_compute_objective=("maximize", "eval_accuracy")),
        stsb=dict(hp_compute_objective=("maximize", "eval_pearson")),
        wnli=dict(
            hp_space=lambda trial: dict(
                learning_rate=tune.loguniform(1e-5, 1e-2),
                max_steps=tune.randint(1, 120),
                warmup_ratio=tune.choice([0., 0.1])),
            hp_num_trials=35,
            hp_compute_objective=("maximize", "eval_accuracy")
        )
    )
)

# ---------
# BERT small trifecta, small tasks
# ---------

hp_search_finetuning_small_bert_trifecta_100k_small_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_small_tasks)
hp_search_finetuning_small_bert_trifecta_100k_small_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_80%_trifecta_100k",  # noqa: E501
)


hp_search_finetuning_small_bert_trifecta_85_100k_small_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_small_tasks)
hp_search_finetuning_small_bert_trifecta_85_100k_small_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_85%_trifecta_100k",  # noqa: E501
)


hp_search_finetuning_small_bert_trifecta_90_100k_small_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_small_tasks)
hp_search_finetuning_small_bert_trifecta_90_100k_small_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_90%_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_2x_100k_small_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_small_tasks)
hp_search_finetuning_small_bert_trifecta_2x_100k_small_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_2x_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_4x_100k_small_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_small_tasks)
hp_search_finetuning_small_bert_trifecta_4x_100k_small_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_4x_trifecta_100k",  # noqa: E501
)

# ---------
# BERT small, big tasks
# ---------

hp_search_finetuning_small_bert_100k_big_tasks = deepcopy(
    debug_finetuning_hp_search)
hp_search_finetuning_small_bert_100k_big_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_large_dataset_100k",  # noqa: E501
    task_name=None,
    task_names=["mnli", "qnli", "qqp", "sst2"],
    eval_steps=2_000,
    hp_space=lambda trial: dict(
        learning_rate=tune.loguniform(1e-6, 1e-3),
        max_steps=tune.randint(20_000, 100_000),
    ),
    hp_num_trials=8,
    task_hyperparams=dict(
        mnli=dict(
            hp_num_trials=5,
            hp_compute_objective=("maximize", "eval_accuracy"),
            eval_steps=4_000,
        ),
        qnli=dict(
            hp_num_trials=5,
            hp_compute_objective=("maximize", "eval_accuracy"),
            eval_steps=4_000,
        ),
        qqp=dict(
            hp_num_trials=5,
            hp_compute_objective=("maximize", "eval_f1"),
            eval_steps=4_000,
        ),
        sst2=dict(
            hp_num_trials=12,
            hp_compute_objective=("maximize", "eval_accuracy"),
            eval_steps=4_000,
        ),
    )
)

hp_search_finetuning_small_bert_100k_qqp = deepcopy(
    hp_search_finetuning_small_bert_100k_big_tasks
)
hp_search_finetuning_small_bert_100k_qqp.update(
    task_names=["qqp"],
    hp_space=lambda trial: dict(
        learning_rate=tune.loguniform(1e-6, 1e-3),
    ),
    max_steps=80_000,
    eval_steps=2_000,
)

# ---------
# BERT small trifecta, big tasks
# ---------

# Assume that learning rate is similar accross tasks within a model
# for large datasets. Therefore, need one sweep per model for big tasks.
hp_search_finetuning_small_bert_trifecta_100k_qqp = deepcopy(
    hp_search_finetuning_small_bert_100k_qqp
)
hp_search_finetuning_small_bert_trifecta_100k_qqp.update(
    task_names=["qqp"],
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_80%_trifecta_100k",  # noqa: E501
    trainer_callbacks=[
        TrackEvalMetrics(),
        RezeroWeightsCallback()]
)

hp_search_finetuning_small_bert_trifecta_85_100k_qqp = deepcopy(
    hp_search_finetuning_small_bert_trifecta_100k_qqp
)
hp_search_finetuning_small_bert_trifecta_85_100k_qqp.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_85%_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_90_100k_qqp = deepcopy(
    hp_search_finetuning_small_bert_trifecta_100k_qqp
)
hp_search_finetuning_small_bert_trifecta_90_100k_qqp.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_90%_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_2x_100k_qqp = deepcopy(
    hp_search_finetuning_small_bert_trifecta_100k_qqp)
hp_search_finetuning_small_bert_trifecta_2x_100k_qqp.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_2x_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_4x_100k_qqp = deepcopy(
    hp_search_finetuning_small_bert_trifecta_100k_qqp)
hp_search_finetuning_small_bert_trifecta_4x_100k_qqp.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_4x_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_100k_big_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_big_tasks)
hp_search_finetuning_small_bert_trifecta_100k_big_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_80%_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_85_100k_big_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_big_tasks)
hp_search_finetuning_small_bert_trifecta_85_100k_big_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_85%_trifecta_100k",  # noqa: E501
)


hp_search_finetuning_small_bert_trifecta_90_100k_big_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_big_tasks)
hp_search_finetuning_small_bert_trifecta_90_100k_big_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_90%_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_2x_100k_big_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_big_tasks)
hp_search_finetuning_small_bert_trifecta_2x_100k_big_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_2x_trifecta_100k",  # noqa: E501
)

hp_search_finetuning_small_bert_trifecta_4x_100k_big_tasks = deepcopy(
    hp_search_finetuning_small_bert_100k_big_tasks)
hp_search_finetuning_small_bert_trifecta_4x_100k_big_tasks.update(
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/small_bert_4x_trifecta_100k",  # noqa: E501
)


# ---------
# BERT Base, small tasks
# ---------


# small tasks refers to tasks with smaller datasets that can be
# run with a larger number of trials
hp_search_finetuning_trifecta_85_100k_small_tasks = deepcopy(
    debug_finetuning_sparse_hp_search)
hp_search_finetuning_trifecta_85_100k_small_tasks.update(
    task_name=None,
    task_names=["cola", "mrpc", "rte", "stsb", "wnli"],
    hp_space=lambda trial: dict(
        learning_rate=tune.loguniform(1e-5, 1e-2),
        max_steps=tune.randint(100, 5000),
        # if warmup 0, no learning rate scheduler
        warmup_ratio=tune.choice([0., 0.1]),
    ),
    hp_num_trials=25,
    task_hyperparams=dict(
        cola=dict(hp_compute_objective=("maximize", "eval_matthews_correlation")),
        mrpc=dict(hp_compute_objective=("maximize", "eval_f1")),
        rte=dict(hp_compute_objective=("maximize", "eval_accuracy")),
        stsb=dict(hp_compute_objective=("maximize", "eval_pearson")),
        wnli=dict(
            hp_space=lambda trial: dict(
                learning_rate=tune.loguniform(1e-5, 1e-2),
                max_steps=tune.randint(1, 120),
                warmup_ratio=tune.choice([0., 0.1])),
            hp_num_trials=35,
            hp_compute_objective=("maximize", "eval_accuracy")
        )
    )
)

# ---------
# BERT base trifecta, small tasks
# ---------

hp_search_finetuning_trifecta_80_100k_small_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_small_tasks)
hp_search_finetuning_trifecta_80_100k_small_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_trifecta_100k"  # noqa
)

hp_search_finetuning_trifecta_90_100k_small_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_small_tasks)
hp_search_finetuning_trifecta_90_100k_small_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_90%_trifecta_100k"  # noqa
)

hp_search_finetuning_bert_100k_small_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_small_tasks
)
hp_search_finetuning_bert_100k_small_tasks.update(
    model_type="bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa
    trainer_callbacks=[TrackEvalMetrics()],
)

hp_search_finetuning_trifecta_2x_small_tasks = deepcopy(
    hp_search_finetuning_trifecta_80_100k_small_tasks)
hp_search_finetuning_trifecta_2x_small_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_2x_trifecta_100k"  # noqa
)

# ---------
# BERT base, big tasks
# ---------

# bigger datasets, small number of trials
hp_search_finetuning_trifecta_85_100k_big_tasks = deepcopy(
    debug_finetuning_sparse_hp_search)
hp_search_finetuning_trifecta_85_100k_big_tasks.update(
    task_name=None,
    task_names=["mnli", "qnli", "qqp", "sst2"],
    eval_steps=2_000,
    hp_space=lambda trial: dict(
        learning_rate=tune.loguniform(1e-6, 1e-3),
        max_steps=tune.randint(20_000, 100_000),
    ),
    hp_num_trials=8,
    task_hyperparams=dict(
        mnli=dict(
            hp_num_trials=5,
            hp_compute_objective=("maximize", "eval_accuracy"),
            eval_steps=4_000,
        ),
        qnli=dict(
            hp_num_trials=5,
            hp_compute_objective=("maximize", "eval_accuracy"),
            eval_steps=4_000,
        ),
        qqp=dict(
            hp_num_trials=5,
            hp_compute_objective=("maximize", "eval_f1"),
            eval_steps=4_000,
        ),
        sst2=dict(
            hp_num_trials=12,
            hp_compute_objective=("maximize", "eval_accuracy"),
            eval_steps=4_000,
        ),
    )
)

# ---------
# BERT base trifecta, big tasks
# ---------

hp_search_finetuning_bert_100k_big_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_big_tasks
)
hp_search_finetuning_bert_100k_big_tasks.update(
    model_type="bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_100k",  # noqa
    trainer_callbacks=[TrackEvalMetrics()],
)

hp_search_finetuning_trifecta_80_100k_big_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_big_tasks)
hp_search_finetuning_trifecta_80_100k_big_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_trifecta_100k"  # noqa
)

hp_search_finetuning_trifecta_85_100k_big_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_big_tasks)
hp_search_finetuning_trifecta_85_100k_big_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_8%_trifecta_100k"  # noqa
)

hp_search_finetuning_trifecta_90_100k_big_tasks = deepcopy(
    hp_search_finetuning_trifecta_85_100k_big_tasks)
hp_search_finetuning_trifecta_90_100k_big_tasks.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_90%_trifecta_100k"  # noqa
)

# The 2x model is going to be too expensive to do hp search all big tasks
hp_search_finetuning_trifecta_2x_qqp = deepcopy(
    hp_search_finetuning_trifecta_80_100k_big_tasks)
hp_search_finetuning_trifecta_2x_qqp.update(
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_2x_trifecta_100k",  # noqa
    task_names=["qqp"],
    task_hyperparams=dict(
        qqp=dict(
            hp_space=lambda trial: dict(
                learning_rate=tune.loguniform(1e-6, 1e-3),
                max_steps=80_000,
                eval_steps=4_000,
            ),
            hp_num_trials=10,
        )
    )
)


# Export configurations in this file
CONFIGS = dict(
    debug_hp_search=debug_hp_search,
    debug_finetuning_hp_search=debug_finetuning_hp_search,
    debug_finetuning_sparse_hp_search=debug_finetuning_sparse_hp_search,
    debug_finetuning_sparse_hp_mnli=debug_finetuning_sparse_hp_mnli,

    # small bert, small tasks
    hp_search_finetuning_small_bert_100k_small_tasks=hp_search_finetuning_small_bert_100k_small_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_100k_small_tasks=hp_search_finetuning_small_bert_trifecta_100k_small_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_85_100k_small_tasks=hp_search_finetuning_small_bert_trifecta_85_100k_small_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_90_100k_small_tasks=hp_search_finetuning_small_bert_trifecta_85_100k_small_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_2x_100k_small_tasks=hp_search_finetuning_small_bert_trifecta_2x_100k_small_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_4x_100k_small_tasks=hp_search_finetuning_small_bert_trifecta_4x_100k_small_tasks,  # noqa: E501

    # small bert, big tasks
    hp_search_finetuning_small_bert_100k_big_tasks=hp_search_finetuning_small_bert_100k_big_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_100k_big_tasks=hp_search_finetuning_small_bert_trifecta_100k_big_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_85_100k_big_tasks=hp_search_finetuning_small_bert_trifecta_85_100k_big_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_90_100k_big_tasks=hp_search_finetuning_small_bert_trifecta_85_100k_big_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_2x_100k_big_tasks=hp_search_finetuning_small_bert_trifecta_2x_100k_big_tasks,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_4x_100k_big_tasks=hp_search_finetuning_small_bert_trifecta_4x_100k_big_tasks,  # noqa: E501

    # small bert, just qqp (instead of all big tasks)
    hp_search_finetuning_small_bert_100k_qqp=hp_search_finetuning_small_bert_100k_qqp,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_100k_qqp=hp_search_finetuning_small_bert_trifecta_100k_qqp,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_85_100k_qqp=hp_search_finetuning_small_bert_trifecta_85_100k_qqp,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_90_100k_qqp=hp_search_finetuning_small_bert_trifecta_90_100k_qqp,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_2x_100k_qqp=hp_search_finetuning_small_bert_trifecta_2x_100k_qqp,  # noqa: E501
    hp_search_finetuning_small_bert_trifecta_4x_100k_qqp=hp_search_finetuning_small_bert_trifecta_4x_100k_qqp,  # noqa: E501

    # bert base, small tasks
    hp_search_finetuning_bert_100k_small_tasks=hp_search_finetuning_bert_100k_small_tasks,  # noqa
    hp_search_finetuning_trifecta_80_100k_small_tasks=hp_search_finetuning_trifecta_80_100k_small_tasks,  # noqa
    hp_search_finetuning_trifecta_85_100k_small_tasks=hp_search_finetuning_trifecta_85_100k_small_tasks,  # noqa
    hp_search_finetuning_trifecta_90_100k_small_tasks=hp_search_finetuning_trifecta_90_100k_small_tasks,  # noqa
    hp_search_finetuning_trifecta_2x_small_tasks=hp_search_finetuning_trifecta_2x_small_tasks,  # noqa

    # bert base, big tasks
    hp_search_finetuning_bert_100k_big_tasks=hp_search_finetuning_bert_100k_big_tasks,
    hp_search_finetuning_trifecta_80_100k_big_tasks=hp_search_finetuning_trifecta_80_100k_big_tasks,  # noqa
    hp_search_finetuning_trifecta_85_100k_big_tasks=hp_search_finetuning_trifecta_85_100k_big_tasks,  # noqa
    hp_search_finetuning_trifecta_90_100k_big_tasks=hp_search_finetuning_trifecta_90_100k_big_tasks,  # noqa
    hp_search_finetuning_trifecta_2x_qqp=hp_search_finetuning_trifecta_2x_qqp,  # noqa
)

# run.py knows to do hyperparameter search instead of finetuning based on
# hp_num_trials. This checks to make sure it is specified in each config.

msg_1 = "hp_num_trials must be > 1 for hyperparameter search"
msg_2 = "hp_num_trials must be specified for hyperparameter search"

for experiment, config in CONFIGS.items():

    if "task_hyperparams" in config:
        hp_config = config["task_hyperparams"]
        for task, task_config in hp_config.items():
            if "hp_num_trials" in task_config:
                n_trials = task_config["hp_num_trials"]
                assert n_trials > 1, msg_1 + f": {experiment}: {task}"
            else:
                full_msg_2 = msg_2 + f": {experiment}: {task}"
                assert config["hp_num_trials"] > 1, full_msg_2

    else:
        assert config["hp_num_trials"] > 1, msg_2 + f": {experiment}"
