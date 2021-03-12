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
    teacher_models_name_or_path=[
        "bert-large-cased",
        # "roberta-large"
    ],
    trainer_extra_kwargs=dict(
        # kd_ensemble_weights=None,
        kd_factor_init=1.0,
        kd_factor_end=1.0,
        kd_temperature_init=1.0,
        kd_temperature_end=1.0,
    )

)

# samples/second (without kd)= 12.066 (on one p3.2xlarge)
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
    # trainer_class=DistillationTrainer,
    # teacher_models_name_or_path=[
    #     "bert-large-cased",
    #     # "roberta-large"
    # ],
    # trainer_extra_kwargs=dict(
    #     # kd_ensemble_weights=None,
    #     kd_factor_init=1.0,
    #     kd_factor_end=1.0,
    #     kd_temperature_init=1.0,
    #     kd_temperature_end=1.0,
    # )
)

# Export configurations in this file
CONFIGS = dict(
    debug_bert_kd=debug_bert_kd,
    tiny_bert_100k_kd=tiny_bert_100k_kd
)
