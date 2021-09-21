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

import unittest
from collections import OrderedDict

import torch

# FIXME: Importing module relative to the project
from projects.transformers.models import SparseEmbeddings


def create_simple_model():
    modules = OrderedDict([
        ("embedding", SparseEmbeddings(torch.nn.Embedding(8, 8), sparsity=0.75)),
        ("linear", torch.nn.Linear(8, 8))
    ])
    return torch.nn.Sequential(modules)


class SparseEmbeddingsTest(unittest.TestCase):
    """
    This is a test for a SparseEmbeddings layer which should behave very similarly to a
    SparseWeights layers.
    """

    def test_forward_and_backward_pass(self):
        model = create_simple_model()
        optim = torch.optim.SGD(model.parameters(), lr=0.01)

        # Validate overall sparsity.
        assert (model.embedding.weight == 0).sum() == 48

        # Validate sparsity per each embedding.
        num_embeddings = model.embedding.module.num_embeddings
        for j in range(num_embeddings):
            embedding_j = model.embedding.weight[j, :]
            assert (embedding_j == 0).sum() == 6

        # Run one forward and backward pass.
        x = torch.Tensor([0, 2]).long()
        y = model(x)
        targets = torch.randint(8, size=(2,))
        loss = torch.nn.functional.cross_entropy(y, targets)
        loss.backward()

        optim.step()
        model.embedding.rezero_weights()

        # Validate overall sparsity.
        assert (model.embedding.weight == 0).sum() == 48

        # Validate sparsity per each embedding.
        num_embeddings = model.embedding.module.num_embeddings
        for j in range(num_embeddings):
            embedding_j = model.embedding.weight[j, :]
            assert (embedding_j == 0).sum() == 6


if __name__ == "__main__":
    unittest.main(verbosity=2)
