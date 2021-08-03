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

from transformers import Trainer

from callbacks import PlotDensitiesCallback, RezeroWeightsCallback
from trainer_mixins import (
    DistillationTrainerMixin,
    GradualMagnitudePruningMixin,
    OneCycleLRMixin,
    ThreeStageLRMixin,
)

from .bert_replication import bert_1mi
from .bertitos import tiny_bert_100k, tiny_bert_debug
from .trifecta import KDLRRangeTestTrainer


class GMPPretrainedTrainer(GradualMagnitudePruningMixin,
                           ThreeStageLRMixin,
                           DistillationTrainerMixin,
                           Trainer):
    """
    GMP on pretrained network. This uses a three stage lr for stabilization, pruning and
    fine-tuning.
    """
    pass


class GMPTrainer(GradualMagnitudePruningMixin,
                 OneCycleLRMixin,
                 DistillationTrainerMixin,
                 Trainer):
    """
    GMP on pretrained network. This uses one-cycle lr.
    """
    pass


# Pretrained models to prune via GMP.
base_dir = "/mnt/efs/results/pretrained-models/transformers-local/"
tiny_bert_100k_pretrained = base_dir + "tiny_bert_100k_prunable"
tiny_bert_kd_onecycle_100k_pretrained = base_dir + "tiny_bert_onecycle_lr_kd_100k_prunable"  # noqa: E501


# Just a debug config.
tiny_bert_gmp_debug = deepcopy(tiny_bert_debug)
tiny_bert_gmp_debug.update(
    max_steps=100,
    do_eval=False,
    model_name_or_path=tiny_bert_100k_pretrained,
    # evaluation_strategy="steps",
    # eval_steps=2,
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(log_steps=10),
    ],
    trainer_class=GMPPretrainedTrainer,
    trainer_mixin_args=dict(
        start_sparsity=0,
        end_sparsity=0.8,
        warmup_steps=30,
        cooldown_steps=30,
        prune_period=10,
        max_lr=0.01,
        verbose_gmp_logging=True,
    ),
)
tiny_bert_gmp_debug["config_kwargs"].update(
    sparsity=0.8,
    sparsify_all_embeddings=False,
)


# ----------
# Tiny BERT
# ----------


# Perform LR range test with pretrained Tiny BERT.
tiny_bert_pretrained_gmp_lr_range_test = deepcopy(tiny_bert_100k)
tiny_bert_pretrained_gmp_lr_range_test.update(
    max_steps=100,

    # LR Range test for `tiny_bert_pretrained_gmp_52k`
    # trainer_class=LRRangeTestTrainer,
    # model_name_or_path=tiny_bert_100k_pretrained,

    # LR Range test for `tiny_bert_kd_onecycle_pretrained_gmp_52k`
    trainer_class=KDLRRangeTestTrainer,
    model_name_or_path=tiny_bert_kd_onecycle_100k_pretrained,

    # eval_steps=1,
    trainer_mixin_args=dict(
        # LR Range Test
        min_lr=1e-8,
        max_lr=5e-5,
        test_mode="linear",

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
    overwrite_output_dir=True,
    do_eval=True,
)


# Apply GMP pruning on pretrained Tiny BERT. Pre-trianing done w/o KD + OneCycle LR
tiny_bert_pretrained_gmp_52k = deepcopy(tiny_bert_100k)
tiny_bert_pretrained_gmp_52k.update(
    # steps: 2000 warmup, 30000 pruning (every 1000), 20000 cooldown
    max_steps=2000 + 30000 + 20000,
    model_type="fully_static_sparse_bert",
    model_name_or_path=tiny_bert_100k_pretrained,
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=10000),
    ],
    trainer_class=GMPPretrainedTrainer,
    trainer_mixin_args=dict(
        # GMP
        start_sparsity=0,
        end_sparsity=0.8,
        warmup_steps=2000,
        cooldown_steps=20000,
        prune_period=1000,
        max_lr=2e-5,
        verbose_gmp_logging=True,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
)


# Apply GMP pruning on pretrained Tiny BERT. Pre-trianing done with KD + OneCycle LR
# LR-Range Test here https://wandb.ai/numenta/huggingface/runs/34iuv3s5
tiny_bert_kd_onecycle_pretrained_gmp_52k = deepcopy(tiny_bert_pretrained_gmp_52k)
tiny_bert_kd_onecycle_pretrained_gmp_52k.update(
    model_name_or_path=tiny_bert_kd_onecycle_100k_pretrained,
)
tiny_bert_kd_onecycle_pretrained_gmp_52k["trainer_mixin_args"].update(
    max_lr=2e-5,
)

tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_002 = deepcopy(tiny_bert_kd_onecycle_pretrained_gmp_52k)  # noqa E501
tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_002["trainer_mixin_args"].update(
    max_lr=2e-3,
)

tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_02 = deepcopy(tiny_bert_kd_onecycle_pretrained_gmp_52k)  # noqa E501
tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_02["trainer_mixin_args"].update(
    max_lr=2e-2,
)

# Apply GMP pruning Tiny BERT throughout pretraining.
tiny_bert_gmp_100k = deepcopy(tiny_bert_100k)
tiny_bert_gmp_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=10000),
    ],
    trainer_class=GMPTrainer,
    trainer_mixin_args=dict(
        # GMP
        start_sparsity=0,
        end_sparsity=0.8,
        warmup_steps=5000,
        cooldown_steps=20000,
        prune_period=500,
        verbose_gmp_logging=True,

        # OneCycle LR
        max_lr=0.03,

        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
)
tiny_bert_gmp_100k["config_kwargs"].update(
    # Only sparsify the word_embeddings
    sparsify_all_embeddings=False,
    sparsity=0,
)


tiny_bert_gmp_100k_maxlr_01 = deepcopy(tiny_bert_gmp_100k)
tiny_bert_gmp_100k_maxlr_01["trainer_mixin_args"].update(
    max_lr=0.01,
)

tiny_bert_gmp_100k_maxlr_05 = deepcopy(tiny_bert_gmp_100k)
tiny_bert_gmp_100k_maxlr_05["trainer_mixin_args"].update(
    max_lr=0.05,
)


# ---------
# BERT Base
# ---------


bert_1mi_pretrained = "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi_prunable"  # noqa E501


# The max_lr for this run is chosen based off this lr-range test
# https://wandb.ai/nupic-research/huggingface/runs/26eavts5 This is for an untrained
# BERT Base, while the config below starts with a pre-trained BERT Base. Nonetheless,
# previous experiments suggest this is a good starting point. Thus, we use a max_lr
# slightly lower than what the test suggests.
bert_1mi_pretrained_gmp_52k = deepcopy(bert_1mi)
bert_1mi_pretrained_gmp_52k.update(
    fp16=True,
    # steps: 2000 warmup, 30000 pruning (every 1000), 20000 cooldown
    max_steps=2000 + 30000 + 20000,  # longer may be better
    model_type="fully_static_sparse_bert",
    model_name_or_path=bert_1mi_pretrained,
    tokenized_data_cache_dir="/mnt/datasets/huggingface/preprocessed-datasets/text",
    overwrite_output_dir=True,
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=10000),
    ],
    trainer_class=GMPPretrainedTrainer,
    trainer_mixin_args=dict(
        # GMP
        start_sparsity=0,
        end_sparsity=0.8,
        warmup_steps=2000,
        cooldown_steps=20000,
        prune_period=1000,
        max_lr=0.0003,
        verbose_gmp_logging=True,
        # KD
        teacher_model_names_or_paths=[
            "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
        ],
    ),
)


CONFIGS = dict(
    # Tiny BERT
    tiny_bert_gmp_debug=tiny_bert_gmp_debug,
    tiny_bert_pretrained_gmp_lr_range_test=tiny_bert_pretrained_gmp_lr_range_test,
    tiny_bert_pretrained_gmp_52k=tiny_bert_pretrained_gmp_52k,
    tiny_bert_kd_onecycle_pretrained_gmp_52k=tiny_bert_kd_onecycle_pretrained_gmp_52k,
    tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_02=tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_02,  # noqa 501
    tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_002=tiny_bert_kd_onecycle_pretrained_gmp_52k_maxlr_002,  # noqa 501
    tiny_bert_gmp_100k=tiny_bert_gmp_100k,
    tiny_bert_gmp_100k_maxlr_01=tiny_bert_gmp_100k_maxlr_01,
    tiny_bert_gmp_100k_maxlr_05=tiny_bert_gmp_100k_maxlr_05,

    # BERT Base
    bert_1mi_pretrained_gmp_52k=bert_1mi_pretrained_gmp_52k,
)
