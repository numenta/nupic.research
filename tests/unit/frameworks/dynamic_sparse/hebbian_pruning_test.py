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
from itertools import product

import torch

# from nupic.research.frameworks.dynamic_sparse.common import *
from nupic.research.frameworks.dynamic_sparse.models import (
    DSNNMixedHeb,
    DSNNWeightedMag,
    SparseModule,
)
from nupic.research.frameworks.dynamic_sparse.networks import MLPHeb


def allclose_boolean(t1, t2):
    return torch.allclose(t1.int(), t2.int())


def expand(list_of_indices):
    return list(zip(*list_of_indices))


class WeightedMagPruningTest(unittest.TestCase):
    def setUp(self):

        self.model = DSNNWeightedMag(network=MLPHeb())
        self.model.setup()

        self.sparse_module = SparseModule(m=torch.nn.Linear(5, 5))

        # dummy coactivation matrix
        self.sparse_module.m.coactivations = torch.tensor(
            [
                [0.3201, 0.8318, 0.3382, 0.9734, 0.0985],
                [0.0401, 0.8620, 0.0845, 0.3778, 0.3996],
                [0.4954, 0.0092, 0.6713, 0.8594, 0.9487],
                [0.8101, 0.0922, 0.2033, 0.7185, 0.4588],
                [0.3897, 0.6865, 0.5072, 0.9749, 0.0597],
            ]
        ).t()
        # the transpose is what will be used
        # ([
        #     [0.3201, 0.0401, 0.4954, 0.8101, 0.3897],
        #     [0.8318, 0.8620, 0.0092, 0.0922, 0.6865],
        #     [0.3382, 0.0845, 0.6713, 0.2033, 0.5072],
        #     [0.9734, 0.3778, 0.8594, 0.7185, 0.9749],
        #     [0.0985, 0.3996, 0.9487, 0.4588, 0.0597]
        # ])

        # dummy weight matrix
        self.weight = torch.tensor(
            [
                [19, 2, -12, 0, 0],
                [0, 0, 0, 0, 0],
                [-10, 25, -8, 0, 0],
                [21, -11, 7, 0, 0],
                [-14, 18, -6, 0, 0],
            ]
        ).float()
        self.sparse_module.m.weight = torch.nn.Parameter(self.weight)
        self.sparse_module.mask = (self.weight != 0).float()
        self.sparse_module.pos = 1
        self.sparse_module.save_num_params()

        # weights
        # [[ 6.0819,  0.0802, -5.9448,  0.0000,  0.0000],
        # [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        # [-3.3820,  2.1125, -5.3704,  0.0000,  0.0000],
        # [20.4414, -4.1558,  6.0158,  0.0000,  0.0000],
        # [-1.3790,  7.1928, -5.6922,  0.0000,  0.0000]]

        self.lowest_50_mag = [(2, 0), (3, 1), (4, 0), (0, 1), (2, 1), (3, 2)]

        # remaining support variables
        self.shape = self.weight.shape
        self.idxs = list(product(*[range(s) for s in self.shape]))
        self.nonzero_idxs = [i for i in self.idxs if self.weight[i] != 0]
        self.zero_idxs = [i for i in self.idxs if self.weight[i] == 0]
        self.num_params = torch.sum(self.weight != 0).item()

    def test_pruning_partial(self):

        self.sparse_module.on_perc = 0.5
        self.sparse_module.hebbian_prune = 0
        self.sparse_module.weight_prune = 0.5

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # test keep mask
        falses = ~keep_mask[expand(self.lowest_50_mag)]
        highest_50 = set(self.nonzero_idxs).difference(self.lowest_50_mag)
        trues = keep_mask[expand(highest_50)]

        self.assertEqual(
            torch.sum(falses).item(), 6, "Lowest 50perc in magnitude should be False"
        )
        self.assertEqual(
            torch.sum(trues).item(), 6, "Highest 50perc in magnitude should be True"
        )

        # test add mask
        new_connections = add_mask[expand(self.zero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            6,
            "Add mask should have 6 previously non-active connections set to True",
        )

        new_connections = add_mask[expand(self.nonzero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            0,
            "Add mask should not impact connections which were previously active",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_pruning_all(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 0
        self.sparse_module.weight_prune = 1

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # keep mask should not include any of previously existing connections
        self.assertEqual(
            torch.sum(keep_mask).item(),
            0,
            "When weight prune perc is 1, keep mask should be all 0s",
        )

        # conversely, the add mask need to have number of elements same as params
        self.assertEqual(
            torch.sum(add_mask).item(),
            self.num_params,
            "When weight prune perc is 1, add mask should replace all params",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_pruning_none(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 0
        self.sparse_module.weight_prune = 0

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # keep mask should not include any of previously existing connections
        self.assertEqual(
            torch.sum(keep_mask).item(),
            self.num_params,
            "When weight prune perc is 0, keep mask should be all 1s",
        )

        # conversely, the add mask need to have number of elements same as params
        self.assertEqual(
            torch.sum(add_mask).item(),
            0,
            "When weight prune perc is 0, add mask should be all 0s",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )


class HebbianPruningTest(unittest.TestCase):
    def setUp(self):

        self.model = DSNNMixedHeb(network=MLPHeb())
        self.model.setup()

        self.sparse_module = SparseModule(m=torch.nn.Linear(5, 5))

        # dummy coactivation matrix
        self.sparse_module.m.coactivations = torch.tensor(
            [
                [0.3201, 0.8318, 0.3382, 0.9734, 0.0985],
                [0.0401, 0.8620, 0.0845, 0.3778, 0.3996],
                [0.4954, 0.0092, 0.6713, 0.8594, 0.9487],
                [0.8101, 0.0922, 0.2033, 0.7185, 0.4588],
                [0.3897, 0.6865, 0.5072, 0.9749, 0.0597],
            ]
        ).t()
        # the transpose is what will be used
        # ([
        #     [0.3201, 0.0401, 0.4954, 0.8101, 0.3897],
        #     [0.8318, 0.8620, 0.0092, 0.0922, 0.6865],
        #     [0.3382, 0.0845, 0.6713, 0.2033, 0.5072],
        #     [0.9734, 0.3778, 0.8594, 0.7185, 0.9749],
        #     [0.0985, 0.3996, 0.9487, 0.4588, 0.0597]
        # ])

        # dummy weight matrix
        self.weight = torch.tensor(
            [
                [19, 2, -12, 0, 0],
                [0, 0, 0, 0, 0],
                [-10, 25, -8, 0, 0],
                [21, -11, 7, 0, 0],
                [-14, 18, -6, 0, 0],
            ]
        ).float()
        self.sparse_module.m.weight = torch.nn.Parameter(self.weight)
        self.sparse_module.mask = (self.weight != 0).float()
        self.sparse_module.pos = 1
        self.sparse_module.save_num_params()

        # manually define which connections will be pruned
        # select lowest 25% correlations
        self.lowest_25_hebb = [(4, 0), (2, 1), (0, 1)]
        # select lowest 50% of negatives and lowest 50% of positives
        self.lowest_50_mag = [(2, 0), (2, 2), (4, 2), (0, 1), (3, 2), (4, 1)]
        self.mag_hebb_intersection = (0, 1)

        # remaining support variables
        self.shape = self.weight.shape
        self.idxs = list(product(*[range(s) for s in self.shape]))
        self.nonzero_idxs = [i for i in self.idxs if self.weight[i] != 0]
        self.zero_idxs = [i for i in self.idxs if self.weight[i] == 0]
        self.num_params = torch.sum(self.weight != 0).item()

    def test_partial_magnitude_and_hebbian(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 0.25
        self.sparse_module.weight_prune = 0.50

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        intersection = self.mag_hebb_intersection
        complement = set(self.nonzero_idxs).difference(intersection)
        trues = keep_mask[expand(complement)]

        self.assertFalse(
            keep_mask[intersection].item(), "Only item in intersection should be False"
        )
        self.assertEqual(
            torch.sum(trues).item(), 11, "All other items should be set to True"
        )

        # test add mask
        new_connections = add_mask[expand(self.zero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            1,
            "Add mask should have 1 previously non-active connections set to True",
        )

        new_connections = add_mask[expand(self.nonzero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            0,
            "Add mask should not impact connections which were previously active",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_partial_magnitude(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 0
        self.sparse_module.weight_prune = 0.5

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # test keep mask
        falses = ~keep_mask[expand(self.lowest_50_mag)]
        highest_50 = set(self.nonzero_idxs).difference(self.lowest_50_mag)
        trues = keep_mask[expand(highest_50)]

        self.assertEqual(
            torch.sum(falses).item(), 6, "Lowest 50perc in magnitude should be False"
        )
        self.assertEqual(
            torch.sum(trues).item(), 6, "Highest 50perc in magnitude should be True"
        )

        # test add mask
        new_connections = add_mask[expand(self.zero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            6,
            "Add mask should have 6 previously non-active connections set to True",
        )

        new_connections = add_mask[expand(self.nonzero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            0,
            "Add mask should not impact connections which were previously active",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_partial_hebbian(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 0.25
        self.sparse_module.weight_prune = 0

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # test keep mask
        falses = ~keep_mask[expand(self.lowest_25_hebb)]
        highest_75 = set(self.nonzero_idxs).difference(self.lowest_25_hebb)
        trues = keep_mask[expand(highest_75)]

        # print(keep_mask)
        self.assertEqual(
            torch.sum(falses).item(), 3, "Lowest 25perc in correlations should be False"
        )
        self.assertEqual(
            torch.sum(trues).item(), 9, "Highest 75perc in correlations should be True"
        )

        # test add mask
        new_connections = add_mask[expand(self.zero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            3,
            "Add mask should have 3 previously non-active connections set to True",
        )

        new_connections = add_mask[expand(self.nonzero_idxs)]
        self.assertEqual(
            torch.sum(new_connections).item(),
            0,
            "Add mask should not impact connections which were previously active",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_full_hebbian(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 1
        self.sparse_module.weight_prune = 0

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # keep mask should not include any of the previously existing connections
        self.assertEqual(
            torch.sum(keep_mask).item(),
            0,
            "When hebbian prune perc is 1, keep mask should be all 0s",
        )

        # conversely, the add mask need to have number of elements same as params
        self.assertEqual(
            torch.sum(add_mask).item(),
            self.num_params,
            "When hebbian prune perc is 1, add mask should replace all params",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_full_magnitude(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 0
        self.sparse_module.weight_prune = 1

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # keep mask should not include any of previously existing connections
        self.assertEqual(
            torch.sum(keep_mask).item(),
            0,
            "When weight prune perc is 1, keep mask should be all 0s",
        )

        # conversely, the add mask need to have number of elements same as params
        self.assertEqual(
            torch.sum(add_mask).item(),
            self.num_params,
            "When weight prune perc is 1, add mask should replace all params",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )

    def test_full_magnitude_and_hebbian(self):

        self.sparse_module.on_perc = 0.1
        self.sparse_module.hebbian_prune = 1
        self.sparse_module.weight_prune = 1

        keep_mask = self.model.prune(self.sparse_module)
        num_params = self.sparse_module.num_params
        num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
        add_mask = self.model.grow(self.sparse_module, num_add)
        new_mask = keep_mask | add_mask

        # keep mask should not include any of the previously existing connections
        self.assertEqual(
            torch.sum(keep_mask).item(),
            0,
            "When weight and hebbian prune perc are 1, keep mask should be all 0s",
        )

        # conversely, the add mask need to have number of elements same as params
        self.assertEqual(
            torch.sum(add_mask).item(),
            self.num_params,
            "If weight and hebbian prune perc are 1, should replace all params",
        )

        # new mask needs to be a combination of both
        dummy_new_mask = keep_mask | add_mask
        self.assertTrue(
            allclose_boolean(new_mask, dummy_new_mask),
            "New mask should be an OR of keep_mask and add_mask",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
