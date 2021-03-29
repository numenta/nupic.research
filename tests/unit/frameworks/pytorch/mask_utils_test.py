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

from nupic.research.frameworks.pytorch.mask_utils import (
    get_topk_submask,
    indices_to_mask,
)


class TestMaskUtils(unittest.TestCase):

    def setUp(self):
        self.tensor = torch.tensor([
            [[0.7076, 0.8986, 0.0251, 0.4676],
             [0.0481, 0.6182, 0.3293, 0.3704],
             [0.0445, 0.5824, 0.7674, 0.8352]],

            [[0.3773, 0.2216, 0.4702, 0.3802],
             [0.2974, 0.2590, 0.0205, 0.3885],
             [0.2964, 0.2388, 0.2593, 0.9136]]
        ])

    def test_indices_to_mask_dim_0(self):

        expected_mask = torch.tensor([
            [[1, 1, 0, 1],
             [0, 1, 1, 0],
             [0, 1, 1, 0]],

            [[0, 0, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1]]
        ])
        indices = self.tensor.max(dim=0).indices
        actual_mask = indices_to_mask(indices, self.tensor.shape, dim=0)
        self.assertTrue(actual_mask.dtype == torch.bool)

        all_equal = (actual_mask == expected_mask).all()
        self.assertTrue(all_equal)

    def test_indices_to_mask_dim_1(self):

        expected_mask = torch.tensor([
            [[1, 1, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 1, 1]],

            [[1, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        ])
        indices = self.tensor.max(dim=1).indices
        actual_mask = indices_to_mask(indices, self.tensor.shape, dim=1)
        self.assertTrue(actual_mask.dtype == torch.bool)

        all_equal = (actual_mask == expected_mask).all()
        self.assertTrue(all_equal)

    def test_indices_to_mask_dim_2(self):

        expected_mask = torch.tensor([
            [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],

            [[0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 1]]
        ])
        indices = self.tensor.max(dim=2).indices
        actual_mask = indices_to_mask(indices, self.tensor.shape, dim=2)
        self.assertTrue(actual_mask.dtype == torch.bool)

        all_equal = (actual_mask == expected_mask).all()
        self.assertTrue(all_equal)

    def test_topk_submask_structure(self):

        values = torch.rand(5, 6, 7)
        k = 5
        mask = torch.rand_like(values) < 0.5
        submask = get_topk_submask(k, values, mask, largest=True)

        # Validate that the submask's on-positions are a subset of the original.
        is_subset = (submask <= mask).all()
        self.assertTrue(is_subset)

        # Validate only k positions have been chosen.
        self.assertEqual(submask.sum(), k)

    def test_get_topk_submask(self):
        k = 3
        values = torch.tensor([
            [0.5086, 0.5467, 0.2095],
            [0.9721, 0.2540, 0.2837],
            [0.4696, 0.9867, 0.6543]
        ])
        mask = torch.tensor([
            [True, True, False],
            [True, True, False],
            [True, False, True]
        ])
        submask = get_topk_submask(k, values, mask, largest=True)
        expected_submask = torch.tensor([
            [False, True, False],
            [True, False, False],
            [False, False, True]
        ])

        # Validate that the submask's on-positions are a subset of the original.
        is_subset = (submask <= mask).all()
        self.assertTrue(is_subset)

        # Validate calculated submask.
        all_equal = (submask == expected_submask).all()
        self.assertTrue(all_equal)

    def test_get_bottomk_submask(self):
        k = 3
        values = torch.tensor([
            [0.5184, 0.1562, 0.3428],
            [0.7742, 0.8507, 0.0986],
            [0.3525, 0.8384, 0.4315]])

        mask = torch.tensor([
            [True, False, True],
            [True, True, False],
            [True, False, False]
        ])
        submask = get_topk_submask(k, values, mask, largest=False)

        expected_submask = torch.tensor([
            [True, False, True],
            [False, False, False],
            [True, False, False]
        ])

        # Validate that the submask's on-positions are a subset of the original.
        is_subset = (submask <= mask).all()
        self.assertTrue(is_subset)

        # Validate calculated submask.
        all_equal = (submask == expected_submask).all()
        self.assertTrue(all_equal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
