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

import numpy as np
import torch
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR, StepLR

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler, ScaledLR


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


class ComposedLRSchedulerTest(unittest.TestCase):
    def test_epoch_update(self):
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=1.0)

        # StepLR(optimizer, step_size=2, gamma=0.1) for first 10 epochs
        expected = np.repeat([1.0, 0.1, 0.01, 0.001, 0.0001], 2).tolist()

        # StepLR(optimizer, step_size=3, gamma=2) for last 15 epochs
        expected += np.repeat([1.0, 2.0, 4.0, 8.0, 16.0], 3).tolist()

        scheduler = ComposedLRScheduler(
            optimizer,
            schedulers={
                0: dict(
                    lr_scheduler_class=StepLR,
                    lr_scheduler_args=dict(step_size=2, gamma=0.1),
                ),
                10: dict(
                    lr_scheduler_class=StepLR,
                    lr_scheduler_args=dict(step_size=3, gamma=2),
                ),
            },
        )
        actual = []
        for epoch in range(len(expected)):
            lr = optimizer.param_groups[0]["lr"]
            actual.append(round(lr, 4))
            # 10 batches
            for batch in range(10):
                optimizer.step()
            # Called once per epoch
            scheduler.step()

        self.assertEqual(expected, actual)

    def test_batch_update(self):

        # OneCycleLR(optimizer, epochs=10, max_lr=2.0, steps_per_epoch=10)
        # for first 10 epochs
        expected = [
            0.08,
            0.5903,
            1.5787,
            1.999,
            1.8806,
            1.5878,
            1.1786,
            0.734,
            0.3421,
            0.0805,
        ]
        # StepLR(optimizer, step_size=3, gamma=0.1) for next 10 epochs
        expected += np.repeat([1.0, 0.1, 0.01, 0.001, 0.0001], 2).tolist()

        # OneCycleLR(optimizer, epochs=10, max_lr=2.0, steps_per_epoch=10)
        # for last 10 epochs
        expected += [
            0.08,
            0.5903,
            1.5787,
            1.999,
            1.8806,
            1.5878,
            1.1786,
            0.734,
            0.3421,
            0.0805,
        ]

        optimizer = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        batch_size = 10
        scheduler = ComposedLRScheduler(
            optimizer,
            steps_per_epoch=batch_size,
            schedulers={
                0: dict(
                    lr_scheduler_class=OneCycleLR,
                    lr_scheduler_args=dict(
                        epochs=10, max_lr=2.0, steps_per_epoch=batch_size
                    ),
                ),
                10: dict(
                    lr_scheduler_class=StepLR,
                    lr_scheduler_args=dict(step_size=2, gamma=0.1),
                ),
                20: dict(
                    lr_scheduler_class=OneCycleLR,
                    lr_scheduler_args=dict(
                        epochs=10, max_lr=2.0, steps_per_epoch=batch_size
                    ),
                ),
            },
        )
        actual = []

        for epoch in range(len(expected)):
            lr = optimizer.param_groups[0]["lr"]
            actual.append(round(lr, 4))
            # 10 batches
            for batch in range(batch_size):
                optimizer.step()
                # Called once per batch
                scheduler.step()

        self.assertEqual(expected, actual)

    def test_optimizer_update(self):
        optimizer = torch.optim.SGD(
            [torch.zeros(1)], lr=1.0, weight_decay=0.0001, momentum=0.0
        )
        batch_size = 10
        expected = [dict(weight_decay=1e-04, momentum=0.0)] * 10
        expected += [dict(weight_decay=1e-05, momentum=0.9)] * 15
        scheduler = ComposedLRScheduler(
            optimizer,
            steps_per_epoch=batch_size,
            schedulers={
                0: dict(
                    lr_scheduler_class=OneCycleLR,
                    lr_scheduler_args=dict(
                        epochs=10,
                        max_lr=2.0,
                        cycle_momentum=False,
                        steps_per_epoch=batch_size,
                    ),
                    optimizer_args=dict(weight_decay=1e-04, momentum=0.0),
                ),
                10: dict(
                    lr_scheduler_class=StepLR,
                    lr_scheduler_args=dict(step_size=2, gamma=0.1),
                    optimizer_args=dict(weight_decay=1e-05, momentum=0.9),
                ),
            },
        )

        actual = []
        for epoch in range(len(expected)):
            params = optimizer.param_groups[0]
            actual.append({k: params[k] for k in ["weight_decay", "momentum"]})
            # 10 batches
            for batch in range(batch_size):
                optimizer.step()
                # Called once per batch
                scheduler.step()

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
