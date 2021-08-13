# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
from copy import deepcopy

from ray import tune
from transformers import Trainer

from callbacks import PlotDensitiesCallback, RezeroWeightsCallback
from trainer_mixins import DistillationTrainerMixin, OneCycleLRMixin, RigLMixin

from .bertitos import tiny_bert_100k
from .finetuning import finetuning_bert700k_glue
from .sparse_bert import fully_static_sparse_bert_100k_fp16
from .sparse_bertitos import small_bert_sparse_100k, tiny_bert_sparse_100k
from .trifecta import KDLRRangeTestTrainer


class RigLOneCycleLRTrainer(OneCycleLRMixin, RigLMixin, Trainer):
    pass


class RigLDistillationTrainer(DistillationTrainerMixin, RigLMixin, Trainer):
    pass


class KDOneCycleLRTrainer(DistillationTrainerMixin, OneCycleLRMixin, Trainer):
    pass


onecycle_args = dict(
    pct_start=0.3,
    anneal_strategy="linear",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25,
    final_div_factor=1e4,
    last_epoch=-1,
)

rigl_args = dict(
    prune_fraction=0.3,
    prune_freq=100,
)

kd_args = dict(
    teacher_model_names_or_paths=[
        "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
    ],
)

# ----------
# Tiny BERT
# ----------

# |------------------------------------------------------------------|
# | model                                 | eval loss      | steps   |
# |---------------------------------------|:--------------:|:-------:|
# | tiny_bert                             | 4.021          | 100k    |
# | tiny_bert_sparse                      | 5.865          | 100k    |
# | tiny_bert_sparse KD + RigL + OneCycle | 3.578          | 100k    |
# | tiny_bert_sparse KD + OneCycle        | 3.827          | 100k    |
# | tiny_bert_sparse                      | 5.774          | 300k    |
# | tiny_bert_sparse KD + RigL + OneCycle | 3.507          | 300k    |
# | tiny_bert_sparse KD + OneCycle        | 3.938          | 300k    |
# |------------------------------------------------------------------|
#

# RigL + OneCycleLR
tiny_bert_rigl_100k_onecycle_lr = deepcopy(tiny_bert_sparse_100k)
tiny_bert_rigl_100k_onecycle_lr.update(
    max_steps=100000,
    trainer_class=RigLOneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.0075,
        **onecycle_args,
        **rigl_args,
    ),
    overwrite_output_dir=True,
    fp16=True,
)


# RigL + KD
tiny_bert_rigl_100k_kd = deepcopy(tiny_bert_sparse_100k)
tiny_bert_rigl_100k_kd.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    trainer_class=RigLDistillationTrainer,
    trainer_mixin_args=dict(
        **kd_args,
        **rigl_args,
    ),
    fp16=True,
    overwrite_output_dir=True,
)


# KD + OneCycleLR
tiny_bert_sparse_300k_onecycle_lr_kd = deepcopy(tiny_bert_sparse_100k)
tiny_bert_sparse_300k_onecycle_lr_kd.update(
    max_steps=300000,
    trainer_class=KDOneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.0075,
        **kd_args,
        **onecycle_args,
    ),
    overwrite_output_dir=True,
    fp16=True,
)


# Dense KD + OneCyle LR
tiny_bert_100k_onecycle_lr_kd = deepcopy(tiny_bert_100k)
tiny_bert_100k_onecycle_lr_kd.update(
    max_steps=100000,
    trainer_class=KDOneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.015,
        **kd_args,
        **onecycle_args,
    ),
    overwrite_output_dir=True,
    fp16=True,
)


tiny_bert_kd_onecycle_lr_range_test = deepcopy(tiny_bert_100k)
tiny_bert_kd_onecycle_lr_range_test.update(
    max_steps=100,
    trainer_class=KDLRRangeTestTrainer,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.005,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)


# KD + OneCycleLR (100k) (eval/loss=4.031)
tiny_bert_sparse_100k_onecycle_lr_kd = deepcopy(tiny_bert_sparse_300k_onecycle_lr_kd)
tiny_bert_sparse_100k_onecycle_lr_kd.update(
    max_steps=100000,
)


# Search for the best max_lr parameters for tiny BERT trained with KD and OneCycle LR
def max_lr_hp_space(trial):
    return dict(
        trainer_mixin_args=dict(
            max_lr=tune.grid_search([
                0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011,
                0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018,
            ]),
        )
    )


tiny_bert_kd_onecycle_50k_maxlr_search = deepcopy(tiny_bert_sparse_300k_onecycle_lr_kd)
tiny_bert_kd_onecycle_50k_maxlr_search.update(
    max_steps=50000,

    # hyperparameter search
    hp_space=max_lr_hp_space,
    hp_num_trials=1,
    hp_validation_dataset_pct=0.05,  # default
    hp_extra_kwargs=dict()  # default
)


# Search for the best pct_start parameters for tiny BERT trained with KD and OneCycle LR
def pct_start_hp_space(trial):
    return dict(
        trainer_mixin_args=dict(
            # Vary percent-start as 10%, 20%, or 30%.
            # The lr will then peak at either 30k, 60k, 90k steps.
            pct_start=tune.grid_search([0.1, 0.2, 0.3]),

            # Use the same max_lr and KD args for each run.
            max_lr=0.01,
            **kd_args,
        )
    )


tiny_bert_kd_onecycle_300k_pct_start_search = deepcopy(tiny_bert_sparse_300k_onecycle_lr_kd)  # noqa E501
tiny_bert_kd_onecycle_300k_pct_start_search.update(
    # hyperparameter search
    hp_space=pct_start_hp_space,
    hp_num_trials=1,
    hp_validation_dataset_pct=0.05,  # default
    hp_extra_kwargs=dict(),  # default

    # Using batch_size of 16 instead of 128 since we're training on 8 GPUs.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
)


tiny_bert_kd_onecycle_100k_pct_start_search = deepcopy(tiny_bert_kd_onecycle_300k_pct_start_search)  # noqa E501
tiny_bert_kd_onecycle_100k_pct_start_search.update(
    max_steps=100000,
)


# ----------
# Small BERT
# ----------


small_bert_rigl_100k_onecycle_lr = deepcopy(small_bert_sparse_100k)
small_bert_rigl_100k_onecycle_lr.update(
    model_type="fully_static_sparse_bert",
    overwrite_output_dir=True,

    # RigL
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    fp16=True,

    # One cycle lr
    trainer_class=RigLOneCycleLRTrainer,
    trainer_mixin_args=dict(
        # One cycle lr
        max_lr=0.003,
        **onecycle_args,
        **rigl_args,
    ),
)


# ---------
# BERT Base
# ---------


# BERT Base with KD + OneCycle LR
# This achieves and eval-loss of 2.28, just slightly over 2.154 from its dense
# counterpart. See `sparse_v4_kd_100k` in the README for more details.
# This took 22h 17m to run on four ps.16xlarges
bert_sparse_100k_kd_oncycle_lr = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_100k_kd_oncycle_lr.update(
    trainer_class=KDOneCycleLRTrainer,
    trainer_mixin_args=dict(

        # One cycle lr
        max_lr=0.0012,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)


# This is an lr-range test for `bert_sparse_100k_kd_oncycle_lr` above.
# This test helped decide to set `max_lr=0.0012`.
# This took 20m to run on four ps.16xlarges
bert_sparse_100k_kd_lr_range_test = deepcopy(fully_static_sparse_bert_100k_fp16)
bert_sparse_100k_kd_lr_range_test.update(
    max_steps=100,
    trainer_class=KDLRRangeTestTrainer,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=0.0001,
        max_lr=0.005,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi"
        ],
    ),
    overwrite_output_dir=True,
)


# This fine-tunes a pretrained model from `bert_sparse_100k_kd_oncycle_lr` above.
# This took 6h 20m to run on a p3.2xlarge
finetuning_bert_sparse_kd_oncycle_lr_100k_glue = deepcopy(finetuning_bert700k_glue)
finetuning_bert_sparse_kd_oncycle_lr_100k_glue.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/bert_sparse_80%_kd_onecycle_lr_100k",  # noqa: E501
)


# This is like the one above, but for 85% sparsity.
bert_sparse_85_kd_lr_range_test = deepcopy(bert_sparse_100k_kd_lr_range_test)
bert_sparse_85_kd_lr_range_test["config_kwargs"].update(
    sparsity=0.85,
)


# This is like the one above, but for 90% sparsity.
bert_sparse_90_kd_lr_range_test = deepcopy(bert_sparse_100k_kd_lr_range_test)
bert_sparse_90_kd_lr_range_test["config_kwargs"].update(
    sparsity=0.90,
    # logging
    overwrite_output_dir=False,
    override_finetuning_results=False,
    task_name=None,
    task_names=["rte", "wnli", "cola"],
    task_hyperparams=dict(
        wnli=dict(num_train_epochs=5, num_runs=20),
        cola=dict(num_train_epochs=5, num_runs=20),
        rte=dict(num_runs=20),
    ),
)

CONFIGS = dict(
    # Tiny BERT
    tiny_bert_rigl_100k_onecycle_lr=tiny_bert_rigl_100k_onecycle_lr,
    tiny_bert_rigl_100k_kd=tiny_bert_rigl_100k_kd,
    tiny_bert_sparse_100k_onecycle_lr_kd=tiny_bert_sparse_100k_onecycle_lr_kd,
    tiny_bert_sparse_300k_onecycle_lr_kd=tiny_bert_sparse_300k_onecycle_lr_kd,
    tiny_bert_100k_onecycle_lr_kd=tiny_bert_100k_onecycle_lr_kd,
    tiny_bert_kd_onecycle_lr_range_test=tiny_bert_kd_onecycle_lr_range_test,
    tiny_bert_kd_onecycle_50k_maxlr_search=tiny_bert_kd_onecycle_50k_maxlr_search,
    tiny_bert_kd_onecycle_100k_pct_start_search=tiny_bert_kd_onecycle_100k_pct_start_search,  # noqa: E501
    tiny_bert_kd_onecycle_300k_pct_start_search=tiny_bert_kd_onecycle_300k_pct_start_search,  # noqa: E501

    # Small BERT
    small_bert_rigl_100k_onecycle_lr=small_bert_rigl_100k_onecycle_lr,

    # BERT Base
    #   80% sparse
    bert_sparse_100k_kd_oncycle_lr=bert_sparse_100k_kd_oncycle_lr,
    bert_sparse_100k_kd_lr_range_test=bert_sparse_100k_kd_lr_range_test,
    finetuning_bert_sparse_kd_oncycle_lr_100k_glue=finetuning_bert_sparse_kd_oncycle_lr_100k_glue,  # noqa: E501
    #   85% sparse
    bert_sparse_85_kd_lr_range_test=bert_sparse_85_kd_lr_range_test,
    #   90% sparse
    bert_sparse_90_kd_lr_range_test=bert_sparse_90_kd_lr_range_test,
)
