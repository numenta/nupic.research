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

import numpy as np
import torch
from transformers import CONFIG_MAPPING, AdamW, AutoModelForMaskedLM

# FIXME: Importing module relative to the project
import projects.transformers.models  # noqa: F401
from nupic.research.frameworks.dynamic_sparse import (
    global_add_by_abs_grad,
    global_prune_by_abs_weight,
)
from nupic.research.frameworks.pytorch.model_utils import (
    count_nonzero_params,
    filter_modules,
)
from nupic.torch.modules.sparse_weights import SparseWeightsBase, rezero_weights


def simple_sparse_transformer():
    config = CONFIG_MAPPING["fully_static_sparse_bert"](
        num_attention_heads=1,
        num_hidden_layers=1,
        hidden_size=4,
        intermediate_size=8,
        max_position_embeddings=10,
        vocab_size=10,
        sparsity=0.75,
    )
    model = AutoModelForMaskedLM.from_config(config)
    return config, model


def calc_sparsity(model):
    tot, nz = count_nonzero_params(model)
    sparsity = 1 - nz / tot
    return sparsity


def init_all_zero_params(model):
    """
    Adjust params so that none are zero. This helps control sparsity levels in case some
    params are randomly zero.
    """
    for p in model.parameters():
        zero_mask = p == 0
        num_zero = zero_mask.sum()
        rand_vals = torch.randn(num_zero) + 1e-5
        p.data[zero_mask] = rand_vals


class GlobalPruningTest(unittest.TestCase):
    def setUp(self):
        self.config, self.model = simple_sparse_transformer()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def test_global_rigl(self):
        """
        Test for globally pruning all sparse modules by their weights and adding back by
        gradients.
        """

        # -----------
        # Init model
        # -----------

        # Make sure there are no random zeros in model params.
        init_all_zero_params(self.model)
        sparsity = calc_sparsity(self.model)
        self.assertEqual(sparsity, 0)

        # Validate initial sparsity after rezeroing the weights.
        self.model.apply(rezero_weights)
        sparsity = calc_sparsity(self.model.bert)
        self.assertTrue(np.isclose(sparsity, 0.4701, atol=1e-4))

        # Get all the SparseWeightsBase modules. These will be pruned.
        sparse_modules = filter_modules(self.model, include_modules=[SparseWeightsBase])
        sparse_modules = sparse_modules.values()
        self.assertEqual(len(sparse_modules), 7)

        # Validate initial number of off params with sparse modules..
        total_sparse_params = np.sum([m.weight.numel() for m in sparse_modules])
        total_off_mask = np.sum([m.zero_mask.bool().sum() for m in sparse_modules])
        total_off_params = np.sum([(m.weight == 0).sum() for m in sparse_modules])
        self.assertEqual(total_sparse_params, 168)
        self.assertEqual(total_off_params, 126)
        self.assertEqual(total_off_mask, 126)

        # --------------
        # Prune weights
        # --------------

        num_removed = global_prune_by_abs_weight(sparse_modules, prune_fraction=1 / 3)

        self.model.apply(rezero_weights)
        total_off_mask = np.sum([m.zero_mask.bool().sum() for m in sparse_modules])
        total_off_params = np.sum([(m.weight == 0).sum() for m in sparse_modules])

        self.assertEqual(total_off_mask, 140)
        self.assertEqual(total_off_params, 140)

        # ---------------
        # Regrow weights
        # ---------------

        # Pseudo forward pass to accumulate gradients.
        batch_size = 2
        num_ebeddings = self.config.max_position_embeddings
        attention_mask = torch.ones(batch_size, num_ebeddings).float()
        input_ids = torch.ones(batch_size, num_ebeddings).long()
        token_type_ids = torch.ones(batch_size, num_ebeddings).long()
        labels = torch.ones(batch_size * num_ebeddings).long()

        outputs = self.model(
            attention_mask=attention_mask,
            input_ids=input_ids,
            labels=labels,
            token_type_ids=token_type_ids,
        )
        loss = outputs.loss
        loss.backward()

        # Add weights according to the largest gradients of the model.
        global_add_by_abs_grad(sparse_modules, num_add=num_removed)

        # The new weights are initialized to zero.
        self.model.apply(rezero_weights)
        total_off_mask = np.sum([m.zero_mask.bool().sum() for m in sparse_modules])
        total_off_params = np.sum([(m.weight == 0).sum() for m in sparse_modules])

        # Validate number of off params after regrowing the weights.
        self.assertEqual(total_off_mask, 126)
        self.assertEqual(total_off_params, 140)

        # Psuedo training step where learning happens on the new zero weights.
        init_all_zero_params(self.model)
        self.model.apply(rezero_weights)

        # Validate number of off params after learning has occurred on new weights.
        total_off_mask = np.sum([m.zero_mask.bool().sum() for m in sparse_modules])
        total_off_params = np.sum([(m.weight == 0).sum() for m in sparse_modules])

        self.assertEqual(total_off_mask, 126)
        self.assertEqual(total_off_params, 126)


if __name__ == "__main__":
    unittest.main(verbosity=2)
