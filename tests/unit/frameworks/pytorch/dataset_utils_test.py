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
from unittest import TestCase

from nupic.research.frameworks.pytorch.dataset_utils import ProgressiveRandomResizedCrop


class ProgressiveRandomResizedCropTest(TestCase):
    def test_set_epoch(self):
        transform = ProgressiveRandomResizedCrop(progressive_resize={
            0: 1,
            2: 2,
            5: 3,
        })
        expected = [1] * 2 + [2] * 3 + [3] * 5
        actual = []
        for epoch in range(10):
            transform.set_epoch(epoch)
            actual.append(transform.image_size)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
