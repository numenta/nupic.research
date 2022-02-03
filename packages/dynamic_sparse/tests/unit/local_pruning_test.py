# Copyright (C) 2022 Numenta Inc. All rights reserved.
#
# The information and source code contained herein is the
# exclusive property of Numenta Inc. No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.

import unittest

import torch
import torch.nn as nn

from nupic.research.frameworks.dynamic_sparse import (
    local_add_by_abs_grad,
    local_prune_by_abs_weight,
)
from nupic.torch.modules import SparseWeights, rezero_weights


def set_weights_to_one(m):
    """
    Set all weights to 1.0 making sure sparse is applied
    """
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(1.0)


class LocalPruningTest(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(
            SparseWeights(nn.Linear(100, 100), sparsity=0.25),
            nn.ReLU(),
            SparseWeights(nn.Linear(100, 100), sparsity=0.75),
            nn.ReLU(),
            nn.Linear(100, 10),
        )
        self.model.apply(set_weights_to_one)
        self.model.apply(rezero_weights)
        self.batch = torch.rand(10, 100)

    def test_local_pruning_and_regrowing(self):
        sparse_modules = [
            m for m in self.model.modules() if isinstance(m, SparseWeights)
        ]

        # Values before pruning
        original_mask = []
        original_nnz = []
        for m in sparse_modules:
            original_mask.append(int(m.zero_mask.bool().sum()))
            original_nnz.append(int(m.weight.bool().sum()))

        # Remove 10% of active weights from each layer
        expected_removed = []
        expected_mask = []
        expected_nnz = []
        for i in range(len(original_nnz)):
            # Expected 10% of nonzero weights to be removed
            removed = round(original_nnz[i] * 0.1)
            expected_removed.append(removed)

            # Expected new amount for nonzero weigths
            nnz = original_nnz[i] - removed
            expected_nnz.append(nnz)

            # Expected new number of zeros in mask
            zeros = original_mask[i] + removed
            expected_mask.append(zeros)

        actual_removed = local_prune_by_abs_weight(sparse_modules, prune_fraction=0.1)
        self.model.apply(rezero_weights)

        # Check returned value
        self.assertListEqual(actual_removed, expected_removed)

        # Check mask
        actual_mask = [int(m.zero_mask.bool().sum()) for m in sparse_modules]
        self.assertListEqual(actual_mask, expected_mask)

        # Check weights
        actual_nnz = [int(m.weight.bool().sum()) for m in sparse_modules]
        self.assertListEqual(actual_nnz, expected_nnz)

        # Accumulate some gradients
        self.model.train()
        y = self.model(self.batch)
        y.mean().backward()

        # Regrow based on gradients restoring the sparsity to the same level as
        # the original model
        local_add_by_abs_grad(sparse_modules, actual_removed)
        self.model.apply(set_weights_to_one)
        self.model.apply(rezero_weights)

        # Check mask
        actual_mask = [int(m.zero_mask.bool().sum()) for m in sparse_modules]
        self.assertListEqual(actual_mask, original_mask)

        # Check weights
        actual_nnz = [int(m.weight.bool().sum()) for m in sparse_modules]
        self.assertListEqual(actual_nnz, original_nnz)
