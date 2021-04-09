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
from trainer_mixins import RigLMixin

from .bertitos import small_bert_100k, tiny_bert_100k, tiny_bert_debug


class RigLTrainer(RigLMixin, Trainer):
    pass


# Results:
# |--------------------------------------------------------------------------|
# | model                        | train loss  | eval loss      | perplexity |
# |------------------------------|:-----------:|:--------------:|:----------:|
# | tiny_bert_static_sparse_300k | 4.537       | 3.997          | 54.432     |
# | tiny_bert_rigl_sparse_300k   | 5.963       | 5.774          | 321.95     |
# |--------------------------------------------------------------------------|
#

# Just a debug config.
tiny_bert_rigl_debug = deepcopy(tiny_bert_debug)
tiny_bert_rigl_debug.update(
    max_steps=10,
    do_eval=True,
    # evaluation_strategy="steps",
    # eval_steps=2,
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=5),
    ],
    trainer_class=RigLTrainer,
    trainer_mixin_args=dict(
        prune_freq=5,
    ),
)
tiny_bert_rigl_debug["config_kwargs"].update(
    sparsity=0.8,
)


# Sparse bert encoding (sparse encoding and sparse embeddings) with RigL
tiny_bert_rigl_100k = deepcopy(tiny_bert_100k)
tiny_bert_rigl_100k.update(
    model_type="static_sparse_encoder_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    trainer_class=RigLTrainer,
    trainer_mixin_args=dict(
        prune_fraction=0.2,
        prune_freq=1000,
    ),
)
tiny_bert_rigl_100k["config_kwargs"].update(
    sparsity=0.5,
)


# Fully static sparse bert (sparse encoding and sparse embeddings).
tiny_bert_static_full_sparse_100k = deepcopy(tiny_bert_100k)
tiny_bert_static_full_sparse_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,
    overwrite_output_dir=True,
)
tiny_bert_static_full_sparse_100k["config_kwargs"].update(
    sparsity=0.8,
)


tiny_bert_static_full_sparse_300k = deepcopy(tiny_bert_static_full_sparse_100k)
tiny_bert_static_full_sparse_300k.update(
    max_steps=300000,
    evaluation_strategy="steps",
    eval_steps=100000,
)


# Fully sparse bert with RigL (dynamic sparse encoding and embeddings)
tiny_bert_full_sparse_rigl_100k = deepcopy(tiny_bert_100k)
tiny_bert_full_sparse_rigl_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    trainer_class=RigLTrainer,
    trainer_mixin_args=dict(
        prune_fraction=0.2,
        prune_freq=1000,
    ),
    fp16=True,
)
tiny_bert_full_sparse_rigl_100k["config_kwargs"].update(
    sparsity=0.8,
)


# This config replicates the RigL paper more closely.
tiny_bert_full_sparse_rigl_300k_prune_perc_30 = deepcopy(tiny_bert_100k)
tiny_bert_full_sparse_rigl_300k_prune_perc_30.update(
    max_steps=300000,
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    trainer_class=RigLTrainer,
    trainer_mixin_args=dict(
        prune_fraction=0.3,
        prune_freq=100,
    ),
    fp16=True,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=100000,
)
tiny_bert_full_sparse_rigl_300k_prune_perc_30["config_kwargs"].update(
    sparsity=0.8,
)


# ----------
# Small BERT
# ----------

small_bert_sparse_100k = deepcopy(small_bert_100k)
small_bert_sparse_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,
    overwrite_output_dir=True,
)

small_bert_rigl_100k = deepcopy(small_bert_100k)
small_bert_rigl_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
        PlotDensitiesCallback(plot_freq=1000),
    ],
    trainer_class=RigLTrainer,
    trainer_mixin_args=dict(
        prune_fraction=0.3,
        prune_freq=100,
    ),
    fp16=True,
    overwrite_output_dir=True,
)


CONFIGS = dict(
    tiny_bert_rigl_debug=tiny_bert_rigl_debug,
    tiny_bert_rigl_100k=tiny_bert_rigl_100k,
    tiny_bert_static_full_sparse_100k=tiny_bert_static_full_sparse_100k,
    tiny_bert_static_full_sparse_300k=tiny_bert_static_full_sparse_300k,
    tiny_bert_full_sparse_rigl_100k=tiny_bert_full_sparse_rigl_100k,
    tiny_bert_full_sparse_rigl_300k_prune_perc_30=tiny_bert_full_sparse_rigl_300k_prune_perc_30,  # noqa E501

    small_bert_sparse_100k=small_bert_sparse_100k,
    small_bert_rigl_100k=small_bert_rigl_100k,
)
