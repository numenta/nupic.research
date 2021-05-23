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

from callbacks import RezeroWeightsCallback

from .bertitos import small_bert_100k, tiny_bert_100k

"""
Sparse versions of Tiny, Mini, and Small BERT. By default, they all train with fp16 and
with 80% BERT sparsity; specifically, with a sparse encoder and sparse word embeddings.
"""

#
# Params and Results:
#
# |---------------------------------------------------------------------|
# | model                   | train loss  | eval loss      | perplexity |
# |-------------------------|:-----------:|:--------------:|:----------:|
# | tiny_bert_sparse_100k   | 6.068       | 5.865          | 352.53     |
# | small_bert_sparse_100k  | 4.193       | 3.805          | 44.943     |
# |---------------------------------------------------------------------|
#


# Tiny BERT with sparse encoder and embeddings.
tiny_bert_sparse_100k = deepcopy(tiny_bert_100k)
tiny_bert_sparse_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,
    overwrite_output_dir=True,
)
tiny_bert_sparse_100k["config_kwargs"].update(
    sparsity=0.8,
    sparsify_all_embeddings=False,
)


# Small BERT (100k) with sparse encoder and embeddings (eval/loss=3.805)
small_bert_sparse_100k = deepcopy(small_bert_100k)
small_bert_sparse_100k.update(
    model_type="fully_static_sparse_bert",
    trainer_callbacks=[
        RezeroWeightsCallback(),
    ],
    fp16=True,
    overwrite_output_dir=True,
)
small_bert_sparse_100k["config_kwargs"].update(
    sparsity=0.8,
    sparsify_all_embeddings=False,
)


# Small BERT (300k) with sparse encoder and embeddings (eval/loss=3.122)
small_bert_sparse_300k = deepcopy(small_bert_sparse_100k)
small_bert_sparse_300k.update(
    max_steps=300000
)


CONFIGS = dict(
    tiny_bert_sparse_100k=tiny_bert_sparse_100k,
    small_bert_sparse_100k=small_bert_sparse_100k,
    small_bert_sparse_300k=small_bert_sparse_300k,
)
