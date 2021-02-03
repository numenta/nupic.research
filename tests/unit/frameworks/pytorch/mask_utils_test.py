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

from nupic.research.frameworks.pytorch.mask_utils import indices_to_mask


class InidicesToMaskTest(unittest.TestCase):

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

        all_equal = (actual_mask == expected_mask).all()
        self.assertTrue(all_equal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
