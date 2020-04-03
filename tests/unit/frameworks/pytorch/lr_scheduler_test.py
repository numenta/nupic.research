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
from torch.optim.lr_scheduler import OneCycleLR, StepLR

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler


class ComposedLRSchedulerTest(unittest.TestCase):
    def test_epoch_update(self):
        num_batches = 15
        optimizer = torch.optim.SGD([torch.zeros(1)], lr=1.0)

        # StepLR(optimizer, step_size=2, gamma=0.1) for first 10 epochs
        expected = np.repeat([1.0, 0.1, 0.01, 0.001, 0.0001], 2).tolist()

        # StepLR(optimizer, step_size=3, gamma=2) for last 15 epochs
        expected += np.repeat([1.0, 2.0, 4.0, 8.0, 16.0], 3).tolist()

        scheduler = ComposedLRScheduler(
            optimizer,
            steps_per_epoch=1,
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
        for _ in expected:
            lr = optimizer.param_groups[0]["lr"]
            actual.append(round(lr, 4))
            for _ in range(num_batches):
                optimizer.step()
            # Called once per epoch
            scheduler.step()

        self.assertEqual(expected, actual)

    def test_batch_update(self):
        num_batches = 15

        # OneCycleLR(optimizer, epochs=10, max_lr=2.0, steps_per_epoch=10)
        # for first 10 epochs, 15 batches per epoch
        expected = [
            0.08,
            0.5799,
            1.559,
            1.9996,
            1.8876,
            1.5998,
            1.1933,
            0.7484,
            0.3534,
            0.0865,
        ]
        # StepLR(optimizer, step_size=3, gamma=0.1) for next 10 epochs
        expected += np.repeat([1.0, 0.1, 0.01, 0.001, 0.0001], 2).tolist()

        # OneCycleLR(optimizer, epochs=10, max_lr=2.0, steps_per_epoch=10)
        # for last 10 epochs, 15 batches per epoch
        expected += [
            0.08,
            0.5799,
            1.559,
            1.9996,
            1.8876,
            1.5998,
            1.1933,
            0.7484,
            0.3534,
            0.0865,
        ]

        optimizer = torch.optim.SGD([torch.zeros(1)], lr=1.0)
        scheduler = ComposedLRScheduler(
            optimizer,
            steps_per_epoch=num_batches,
            schedulers={
                0: dict(
                    lr_scheduler_class=OneCycleLR,
                    lr_scheduler_args=dict(
                        epochs=10, max_lr=2.0, steps_per_epoch=num_batches
                    ),
                ),
                10: dict(
                    lr_scheduler_class=StepLR,
                    lr_scheduler_args=dict(step_size=2, gamma=0.1),
                ),
                20: dict(
                    lr_scheduler_class=OneCycleLR,
                    lr_scheduler_args=dict(
                        epochs=10, max_lr=2.0, steps_per_epoch=num_batches
                    ),
                ),
            },
        )
        actual = []

        for _ in expected:
            lr = optimizer.param_groups[0]["lr"]
            actual.append(round(lr, 4))
            for _ in range(num_batches):
                optimizer.step()
                # Called once per batch
                scheduler.step()

        self.assertEqual(expected, actual)

    def test_optimizer_update(self):
        optimizer = torch.optim.SGD(
            [torch.zeros(1)], lr=1.0, weight_decay=0.0001, momentum=0.0
        )
        num_batches = 15
        expected = [dict(weight_decay=1e-04, momentum=0.0)] * 10
        expected += [dict(weight_decay=1e-05, momentum=0.9)] * 15
        scheduler = ComposedLRScheduler(
            optimizer,
            steps_per_epoch=num_batches,
            schedulers={
                0: dict(
                    lr_scheduler_class=OneCycleLR,
                    lr_scheduler_args=dict(
                        epochs=10,
                        max_lr=2.0,
                        cycle_momentum=False,
                        steps_per_epoch=num_batches,
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
        for _ in range(len(expected)):
            params = optimizer.param_groups[0]
            actual.append({k: params[k] for k in ["weight_decay", "momentum"]})
            for _ in range(num_batches):
                optimizer.step()
                # Called once per batch
                scheduler.step()

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
