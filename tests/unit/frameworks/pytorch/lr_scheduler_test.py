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
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR

from nupic.research.frameworks.pytorch.lr_scheduler import ScaledLR


class ScaledLRTest(unittest.TestCase):

    def test_chaining_schedulers(self):
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        lr_scheduler1 = MultiStepLR(optimizer=optimizer,
                                    milestones=[5, 10, 15, 20],
                                    gamma=0.1)
        lr_scheduler2 = ScaledLR(optimizer=optimizer,
                                 lr_scale={
                                     0: 1.0,
                                     5: 2.0,
                                     10: 3.0,
                                     15: 4.0,
                                     20: 5.0,
                                 })
        expected = [1.] * 5 + [.2] * 5 + [.03] * 5 + [.004] * 5 + [.0005] * 5
        actual = []
        for _ in range(len(expected)):
            lr = optimizer.param_groups[0]["lr"]
            actual.append(round(lr, 8))
            optimizer.step()
            lr_scheduler1.step()
            lr_scheduler2.step()

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
