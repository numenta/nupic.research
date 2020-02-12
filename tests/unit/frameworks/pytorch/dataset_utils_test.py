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

import torch

from nupic.research.frameworks.pytorch.dataset_utils import (
    FakeDataLoader,
    ProgressiveRandomResizedCrop,
)


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

    def test_fake_data_loader(self):

        dataloader = FakeDataLoader(
            dataset_size=12,
            batch_size=3,
            image_size=(1, 28, 28)
        )

        batches = 0
        for image, target in dataloader:
            batches += 1
            self.assertTrue(image.shape == torch.Size([3, 1, 28, 28]))
            self.assertTrue(target.shape == torch.Size([3]))

        self.assertTrue(batches == 4)


if __name__ == "__main__":
    unittest.main()
