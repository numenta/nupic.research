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

import torch

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.research.frameworks.pytorch.models.common_models import SparseMLP


class SparseMLPTests(unittest.TestCase):
    def test_forward_output_shape(self):
        """Validate shape of forward output."""
        input_dim = 10
        output_dim = 11

        sparse_mlp = SparseMLP(input_dim, output_dim)

        batch_size = 8
        x = torch.rand(batch_size, input_dim)

        out = sparse_mlp(x)
        self.assertEqual(out.shape, (batch_size, output_dim))

    def test_segment_sparsity(self):
        """Test sparse mlp hidden layer."""
        input_dim = 10
        hidden_dim = 100
        sparse_mlp = SparseMLP(input_dim,
                               11,
                               hidden_sizes=(hidden_dim,),
                               weight_sparsity=(0.1,)
                               )

        params, nonzero_params = count_nonzero_params(sparse_mlp.linear1)
        self.assertEqual(200, nonzero_params)  # 200 since there is a nonzero bias


if __name__ == "__main__":
    unittest.main(verbosity=2)
