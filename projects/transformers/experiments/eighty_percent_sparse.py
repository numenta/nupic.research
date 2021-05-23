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

from .finetuning import finetuning_bert700k_glue
from .trifecta import small_bert_trifecta_100k, tiny_bert_trifecta_100k

"""
Up till now, many configs are meant to be 80% sparse, but fall short. This is
due to two reasons:
    1. The LayerNorm parameters are dense
    2. The token and position embeddings are dense

This first set will remain dense while the second set can be made sparse. These
experiments attempt to sparsify these embeddings (e.x. `all_embeddings_sparse`)
and/or increase the sparsity among other parameters to get the full 80% sparse.
"""

# ---------
# Tiny BERT
# ---------


# Make all embeddings sparse, including the position and token embeddings.
# Note that the word embeddings were already sparse in `tiny_bert_trifecta_100k`
tiny_bert_trifecta_all_embeddings_sparse = deepcopy(tiny_bert_trifecta_100k)
tiny_bert_trifecta_all_embeddings_sparse["config_kwargs"].update(
    sparsify_all_embeddings=True
)

# Increase the sparsity. The resulting BERT sparsity (given the dense LayerNorm params)
# is should be 80.06%
tiny_bert_trifecta_801_sparse = deepcopy(tiny_bert_trifecta_100k)
tiny_bert_trifecta_801_sparse["config_kwargs"].update(
    sparsity=0.801
)


# Try both approaches: Sparsify all embeddings and increase the sparsity of the model.
tiny_bert_trifecta_801_all_sparse = deepcopy(tiny_bert_trifecta_100k)
tiny_bert_trifecta_801_all_sparse["config_kwargs"].update(
    sparsify_all_embeddings=True,
    sparsity=0.801,
)


# This fine-tunes a pretrained model from `tiny_bert_trifecta_all_embeddings_sparse`
# above.
finetuning_tiny_bert_trifecta_sparse_embeddings_100k = deepcopy(finetuning_bert700k_glue)  # noqa: E501
finetuning_tiny_bert_trifecta_sparse_embeddings_100k.update(
    # Model arguments
    model_type="fully_static_sparse_bert",
    model_name_or_path="/mnt/efs/results/pretrained-models/transformers-local/tiny_bert_trifecta_sparse_embeddings_100k",  # noqa: E501
)
finetuning_tiny_bert_trifecta_sparse_embeddings_100k["config_kwargs"] = dict(
    sparsify_all_embeddings=True,
)


# ----------
# Small BERT
# ----------

# Try both approaches: Sparsify all embeddings and increase the sparsity of the model.
small_bert_trifecta_801_all_sparse = deepcopy(small_bert_trifecta_100k)
small_bert_trifecta_801_all_sparse["config_kwargs"].update(
    sparsify_all_embeddings=True,
    sparsity=0.801,
)


CONFIGS = dict(
    # Tiny BERT
    tiny_bert_trifecta_801_all_sparse=tiny_bert_trifecta_801_all_sparse,
    tiny_bert_trifecta_all_embeddings_sparse=tiny_bert_trifecta_all_embeddings_sparse,
    tiny_bert_trifecta_801_sparse=tiny_bert_trifecta_801_sparse,
    finetuning_tiny_bert_trifecta_sparse_embeddings_100k=finetuning_tiny_bert_trifecta_sparse_embeddings_100k,  # noqa: E501

    # Small BERT
    small_bert_trifecta_801_all_sparse=small_bert_trifecta_801_all_sparse,
)
