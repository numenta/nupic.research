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

import numpy as np
import torch

from nupic.torch.modules.sparse_weights import SparseWeightsBase

__all__ = ["SparseEmbeddings"]


class SparseEmbeddings(SparseWeightsBase):
    """
    This wraps a torch.nn.Embedding module to sparsify the weights where the sparsity is
    applied per embedding. The embedding of an arbitrary index j will have the desired
    sparsity specified through the init.

    Note: A torch.nn.Embedding is already sparse in one sense. Specifically, it's input
    is expected to be sparse (i.e. an integer specifying the index of the embedding).
    In contrast, this introduces sparsity in the weights of the embedding layer, which
    effectively yields sparse output embeddings.

    :param module: A torch.nn.Embedding module
    :param sparsity: Sparsity to apply to the weights; each output embedding will have
                     this level of sparsity.
    """

    def __init__(self, module, sparsity=None):
        assert len(module.weight.shape) == 2, "Should resemble a nn.Embedding"
        super(SparseEmbeddings, self).__init__(
            module, sparsity=sparsity
        )

        # For each unit, decide which weights are going to be zero
        num_embeddings = self.module.num_embeddings
        embedding_dim = self.module.embedding_dim
        num_nz = int(round((1 - self.sparsity) * embedding_dim))
        zero_mask = torch.ones(num_embeddings, embedding_dim, dtype=torch.bool,
                               device=module.weight.device)
        for embedding_j in range(num_embeddings):
            on_indices = np.random.choice(embedding_dim, num_nz, replace=False)
            zero_mask[embedding_j, on_indices] = False

        self.register_buffer("zero_mask", zero_mask)

        self.rezero_weights()

    def rezero_weights(self):
        self.module.weight.data[self.zero_mask] = 0
