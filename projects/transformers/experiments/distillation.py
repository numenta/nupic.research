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

from trainer_mixins import DistillationTrainerMixin

from .base import debug_bert
from .bert_replication import bert_100k


class DistillationTrainer(DistillationTrainerMixin, Trainer):
    pass


debug_bert_kd = deepcopy(debug_bert)
debug_bert_kd.update(

    #  Data Training arguments
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",

    # Training Arguments
    logging_steps=30,
    warmup_steps=30,
    max_steps=300,
    overwrite_output_dir=True,

    # Distillation Arguments
    trainer_class=DistillationTrainer,

    mixin_args=dict(
        # kd_ensemble_weights=None,
        teacher_model_names_or_paths=["bert-large-cased"],
        teacher_model_cache_dir="/mnt/efs/results/pretrained-models/huggingface",
        kd_factor_init=1.0,
        kd_factor_end=1.0,
        kd_temperature_init=1.0,
        kd_temperature_end=1.0,
    )

)

# expected ppl: 55.79276
# results: 90.8664 (distillation from bert base cased from HF library)
# train loss: 4.4652 (distillation loss), eval loss: 4.5094
# 2.25 steps/sec
tiny_bert_100k_kd = deepcopy(bert_100k)
tiny_bert_100k_kd.update(
    # Model Arguments
    model_type="bert",
    config_kwargs=dict(
        num_attention_heads=2,
        num_hidden_layers=2,
        hidden_size=128,  # hidden_size = 64 * num_attention_heads
        intermediate_size=512,  # intermediate_size = 4 * hidden_size
        max_position_embeddings=128,
    ),
    # Training
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    # Dataset
    max_seq_length=128,
    dataset_name="wikipedia_plus_bookcorpus",
    dataset_config_name=None,

    # Distillation Arguments
    trainer_class=DistillationTrainer,
    mixin_args=dict(
        # kd_ensemble_weights=None,
        teacher_model_names_or_paths=[
            "bert-base-cased"
            # "prajjwal1/bert-tiny",
            # "roberta-large"
        ],
        teacher_model_cache_dir="/mnt/efs/results/pretrained-models/huggingface",
        kd_factor_init=1.0,
        kd_factor_end=1.0,
        kd_temperature_init=1.0,
        kd_temperature_end=1.0,
    ),
    overwrite_output_dir=True,

)

# expected ppl: 330.234
# results: 40.817084 (distillation from bert base cased from HF library)
# train loss: 4.1824 (distillation loss), eval loss: 5.3511
# 2.25 steps/sec
tiny_bert_50k_kd = deepcopy(tiny_bert_100k_kd)
tiny_bert_50k_kd.update(
    teacher_models_name_or_path=[
        "/mnt/efs/results/pretrained-models/transformers-local/bert_1mi",
    ],
    overwrite_output_dir=True,

    # Training Arguments
    learning_rate=1e-2,  # increased from 1e-4 to 1e-2
    # min_lr_ratio not an argument to TrainingArguments
    adam_beta1=0.9,  # default
    adam_beta2=0.999,  # default
    adam_epsilon=1e-8,  # default
    weight_decay=1e-7,  # lowered from 1e-2 to 1e-7
    warmup_steps=1000,
    max_steps=50000,
    lr_scheduler_type="linear",
)


def hp_space_lr_search_loguniform(trial):
    return dict(
        learning_rate=tune.grid_search([3, 1, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3])
    )


tiny_bert_50k_kd_lrsearch = deepcopy(tiny_bert_50k_kd)
tiny_bert_50k_kd_lrsearch.update(
    learning_rate=1e-2,

    # hyperparameter search
    hp_space=hp_space_lr_search_loguniform,  # required
    hp_num_trials=1,
    hp_validation_dataset_pct=0.1,  # default
    hp_extra_kwargs=dict()  # default

)


# Export configurations in this file
CONFIGS = dict(
    debug_bert_kd=debug_bert_kd,
    tiny_bert_100k_kd=tiny_bert_100k_kd,
    tiny_bert_50k_kd=tiny_bert_50k_kd,
    tiny_bert_50k_kd_lrsearch=tiny_bert_50k_kd_lrsearch,
)
