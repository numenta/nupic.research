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
Finetuning on Squad, not Glue.
"""

from copy import deepcopy

from transformers import Trainer

from .finetuning import finetuning_bert_100k_glue_get_info
from callbacks import RezeroWeightsCallback, TrackEvalMetrics
from trainer_mixins import QuestionAnsweringMixin

class QuestionAnsweringTrainer(QuestionAnsweringMixin, Trainer):
    pass


debug_bert_squad = deepcopy(finetuning_bert_100k_glue_get_info)
debug_bert_squad.update(
    finetuning=True,
    task_names=None,
    task_name="squad",
    dataset_name="squad",
    dataset_config_name="plain_text",
    trainer_class=QuestionAnsweringTrainer,
    model_name_or_path="bert-base-cased",
    max_seq_length=128,
    do_train=True,
    do_eval=True,
    do_predict=True,
    trainer_callbacks=[
        TrackEvalMetrics(),
    ],
    max_steps=100,
    eval_steps=20,
    rm_checkpoints=True,
    load_best_model_at_end=True,
)

# Export configurations in this file
CONFIGS = dict(
    debug_bert_squad=debug_bert_squad
)
