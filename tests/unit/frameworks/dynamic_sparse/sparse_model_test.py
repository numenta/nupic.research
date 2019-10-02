#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import unittest

import torch

from nupic.research.frameworks.dynamic_sparse.models import SparseModel
from nupic.research.frameworks.dynamic_sparse.networks import MLPHeb, gsc_sparse_dsnn


def has_params(layer):
    return "weight" in layer._parameters


def count_params(network):
    total_params = 0
    for layer in network.modules():
        if has_params(layer):
            nonzeros = torch.nonzero(layer.weight.data)
            total_params += len(nonzeros)

    return total_params


class SparseModelTest(unittest.TestCase):

    def setUp(self):
        self.network1 = MLPHeb(
            config=dict(
                input_size=300,
                num_classes=10,
                hidden_sizes=[100, 100, 100],
                bias=False,
                batch_norm=False,
            )
        )

        self.network2 = gsc_sparse_dsnn({})

    def test_initialize_sparsify_fixed(self):

        network = self.network1

        params_before = count_params(network)

        on_perc = 0.1
        model = SparseModel(
            network=network, config=dict(sparsify_fixed=True, on_perc=on_perc)
        )
        model.setup()
        params_after = count_params(network)

        self.assertEqual(
            int(params_before * on_perc),
            params_after,
            "Number of params should be equal to total params * on perc",
        )

    def test_initialize_sparsify_stochastic(self):
        network = self.network1

        params_before = count_params(network)

        on_perc = 0.1
        threshold = on_perc * 0.25 * params_before

        model = SparseModel(
            network=network, config=dict(sparsify_fixed=False, on_perc=on_perc)
        )
        model.setup()
        params_after = count_params(network)

        self.assertAlmostEqual(
            params_before * on_perc,
            params_after,
            delta=threshold,
            msg="Number of params should be approximate to total params * on perc",
        )

    def test_sparse_modules_count(self):

        network = self.network1
        on_perc = 0.1

        model = SparseModel(
            network=network, config=dict(sparsify_fixed=False, on_perc=on_perc)
        )
        model.setup()

        sparse_modules1 = model.sparse_modules
        self.assertTrue(len(sparse_modules1) == 4)
        ds_modules_1 = model.dynamic_sparse_modules
        self.assertTrue(len(ds_modules_1) == 4)
        self.assertTrue(set(ds_modules_1) <= set(sparse_modules1))

        network = self.network2
        on_perc = 0.1

        model = SparseModel(
            network=network, config=dict(sparsify_fixed=False, on_perc=on_perc)
        )
        model.setup()

        sparse_modules2 = model.sparse_modules
        self.assertTrue(len(sparse_modules2) == 4)
        ds_modules_2 = model.dynamic_sparse_modules
        self.assertTrue(len(ds_modules_2) == 0)
        self.assertTrue(set(ds_modules_2) <= set(sparse_modules2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
