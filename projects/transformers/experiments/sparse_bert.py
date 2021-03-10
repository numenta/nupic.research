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

from callbacks import RezeroWeightsCallback

from .base import debug_bert
from .bert_replication import bert_100k

sparse_bert_100k = deepcopy(bert_100k)
sparse_bert_100k.update(
    # run_name is optional, gets name from experiment name when not defined
    run_name="bert-steps_100k-sparsity_0.8",
    # Model Arguments
    model_type="static_sparse_non_attention_bert",
    config_kwargs=dict(
        sparsity=0.8,
    ),
    trainer_callbacks=[RezeroWeightsCallback()],
    overwrite_output_dir=False,
)


static_sparse_encoder_bert_100k = deepcopy(bert_100k)
static_sparse_encoder_bert_100k.update(
    # Model Arguments
    overwrite_output_dir=False,
    model_type="static_sparse_encoder_bert",
    config_kwargs=dict(
        sparsity=0.8,
    ),
    trainer_callbacks=[RezeroWeightsCallback()]
)


fully_static_sparse_bert_100k = deepcopy(bert_100k)
fully_static_sparse_bert_100k.update(
    # Model Arguments
    overwrite_output_dir=False,
    model_type="fully_static_sparse_bert",
    config_kwargs=dict(
        sparsity=0.8,
    ),
    trainer_callbacks=[RezeroWeightsCallback()]
)


# Sparse Bert of only two layers and one attention head.
mini_sparse_bert_debug = deepcopy(debug_bert)
mini_sparse_bert_debug.update(
    model_type="static_sparse_encoder_bert",
    config_kwargs=dict(
        num_hidden_layers=2,
        num_attention_heads=1,
        sparsity=0.9,
        hidden_size=64,
        intermediate_size=64 * 4,
    ),
    trainer_callbacks=[RezeroWeightsCallback()],
)


# Export configurations in this file
CONFIGS = dict(
    sparse_bert_100k=sparse_bert_100k,
    mini_sparse_bert_debug=mini_sparse_bert_debug,
    static_sparse_encoder_bert_100k=static_sparse_encoder_bert_100k,
    fully_static_sparse_bert_100k=fully_static_sparse_bert_100k,
)
