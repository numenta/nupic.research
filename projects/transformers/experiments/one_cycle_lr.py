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

from trainer_mixins import LRRangeTestMixin, OneCycleLRMixin

from .bertitos import tiny_bert_50k, tiny_bert_debug
from .sparse_bertitos import tiny_bert_sparse_100k

"""
Experiment to train BERT models with OneCycle LR.

Note: it seems easier to encounter exploding gradients when
    * lr is large (order of magnitude 1.0)
    * using fp16 at moderately high lr's (order of magnitude 0.1)
"""

# Results:
# |------------------------------------------------------------------------|
# | model                      | train loss  | eval loss      | perplexity |
# |----------------------------|:-----------:|:--------------:|:----------:|
# | tiny_bert_50k              | 5.990       | 5.800          | 330.234    |
# | tiny_bert_one_cycle_lr_50k | 4.083       | 3.605          | 36.767     |
# |------------------------------------------------------------------------|
#


class OneCycleLRTrainer(OneCycleLRMixin, Trainer):
    pass


class LRRangeTestTrainer(LRRangeTestMixin, Trainer):
    pass


# Just a debug config.
tiny_bert_one_cycle_lr_debug = deepcopy(tiny_bert_debug)
tiny_bert_one_cycle_lr_debug.update(
    max_steps=10,
    do_eval=False,

    trainer_class=OneCycleLRTrainer,
    model_type="bert",
    logging_steps=1,
    logging_first_step=True,

    trainer_mixin_args=dict(
        max_lr=1.0,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,
    ),
    learning_rate=0.1,
)


# Train with one cycle based of the lr found in the range tests below.
# Note, it seems easier to observe exploding gradients when
#    * lr is large (order of magnitude 1.0)
#    * using fp16 at moderately high lr's (order of magnitude 0.1)
tiny_bert_one_cycle_lr_50k = deepcopy(tiny_bert_50k)
tiny_bert_one_cycle_lr_50k.update(

    trainer_class=OneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.01,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,
    ),
    overwrite_output_dir=True,
)


# Train an 80% sparse Tiny Bert with OneCycle LR.
tiny_bert_sparse_100k_onecycle_lr = deepcopy(tiny_bert_sparse_100k)
tiny_bert_sparse_100k_onecycle_lr.update(

    trainer_class=OneCycleLRTrainer,
    trainer_mixin_args=dict(
        max_lr=0.0075,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,
    ),
    overwrite_output_dir=True,
)


# --------------
# LR Range Test
# --------------


# lr range test with linear ramp up
tiny_bert_linear_lr_range_test = deepcopy(tiny_bert_50k)
tiny_bert_linear_lr_range_test.update(

    max_steps=100,
    overwrite_output_dir=True,
    do_eval=True,
    trainer_class=LRRangeTestTrainer,
    trainer_mixin_args=dict(
        min_lr=1e-5,
        max_lr=0.5,
        test_mode="linear"
    ),
)


# lr range test with exponential ramp up
tiny_bert_exponential_lr_range_test = deepcopy(tiny_bert_linear_lr_range_test)
tiny_bert_exponential_lr_range_test["trainer_mixin_args"].update(
    test_mode="exponential"
)


CONFIGS = dict(
    tiny_bert_one_cycle_lr_debug=tiny_bert_one_cycle_lr_debug,
    tiny_bert_one_cycle_lr_50k=tiny_bert_one_cycle_lr_50k,
    tiny_bert_sparse_100k_onecycle_lr=tiny_bert_sparse_100k_onecycle_lr,

    # LR Range Tests
    tiny_bert_linear_lr_range_test=tiny_bert_linear_lr_range_test,
    tiny_bert_exponential_lr_range_test=tiny_bert_exponential_lr_range_test,
)
