# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

"""
Unit tests to ensure that target variables are being transformed properly according to
the continual learning setup found in the benchmarks repository:
https://github.com/GMvandeVen/continual-learning
"""

import unittest

from torch.utils.data import DataLoader
from torchvision import transforms

from nupic.research.frameworks.continuous_learning.multihead.multihead import (
    get_active_classes,
)
from nupic.research.frameworks.pytorch.dataset_utils.dataset_utils import split_dataset
from nupic.research.frameworks.pytorch.test_utils.fake_data import (
    FakeDataPredefinedTargets,
)


def setup(scenario):
    fake_train_dataset = FakeDataPredefinedTargets(image_size=(1, 28, 28))
    fake_test_dataset = FakeDataPredefinedTargets(image_size=(1, 28, 28))

    if scenario in ("task", "domain"):
        target_transform = transforms.Lambda(lambda y: y % 2)
    else:
        target_transform = None

    train_datasets = split_dataset(fake_train_dataset,
                                   groupby=lambda x: x[1] // 2,
                                   target_transform=target_transform)
    test_datasets = split_dataset(fake_test_dataset,
                                  groupby=lambda x: x[1] // 2,
                                  target_transform=target_transform)

    train_loaders = [DataLoader(train_dataset, batch_size=256, shuffle=True)
                     for train_dataset in train_datasets]
    test_loaders = [DataLoader(test_dataset, batch_size=500, shuffle=True)
                    for test_dataset in test_datasets]

    return train_loaders, test_loaders


class MultiHeadTest(unittest.TestCase):

    def test_task_scenario(self):
        """
        Test all target values to be in {0, 1} in the task scenario.
        """
        train_loaders, test_loaders = setup("task")
        for train_loader in train_loaders:
            for _data, target in train_loader:
                self.assertLess(target.max().item(), 2)
                self.assertGreaterEqual(target.min().item(), 0)
        for test_loader in test_loaders:
            for _data, target in test_loader:
                self.assertLess(target.max().item(), 2)
                self.assertGreaterEqual(target.min().item(), 0)

    def test_domain_scenario(self):
        """
        Test all target values to be in {0, 1} in the domain scenario.
        """
        train_loaders, test_loaders = setup("domain")
        for train_loader in train_loaders:
            for _data, target in train_loader:
                self.assertLess(target.max().item(), 2)
                self.assertGreaterEqual(target.min().item(), 0)
        for test_loader in test_loaders:
            for _data, target in test_loader:
                self.assertLess(target.max().item(), 2)
                self.assertGreaterEqual(target.min().item(), 0)

    def test_class_scenario(self):
        """
        Test all target values to be changing based on the task number in the class
        scenario.
        """
        train_loaders, test_loaders = setup("class")
        for task_num, train_loader in enumerate(train_loaders):
            min_target, max_target = 2 * task_num, 2 * task_num + 1
            for _data, target in train_loader:
                # print(target, min_target, max_target)
                try:
                    self.assertLess(target.max().item(), max_target + 1)
                    self.assertGreaterEqual(target.min().item(), min_target)
                except AssertionError:
                    pass
        for task_num, test_loader in enumerate(test_loaders):
            min_target, max_target = 2 * task_num, 2 * task_num + 1
            for _data, target in test_loader:
                self.assertLess(target.max().item(), max_target + 1)
                self.assertGreaterEqual(target.min().item(), min_target)

    def test_get_active_classes_task(self):
        """
        Test the active classes for the task scenario to be changing based on the task
        number in the task scenario.
        """
        active_classes = get_active_classes(3, "task")
        self.assertEqual(active_classes, [4, 5])

    def test_get_active_classes_domain(self):
        """
        Test the active classes for the domain scenario to always be None.
        """
        active_classes = get_active_classes(5, "domain")
        self.assertIsNone(active_classes)

    def test_get_active_classes_class(self):
        """
        Test the active classes for the class scenario to be an enumeration of the
        current and previous labels the model encountered.
        """
        active_classes = get_active_classes(2, "class")
        self.assertEqual(active_classes, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
