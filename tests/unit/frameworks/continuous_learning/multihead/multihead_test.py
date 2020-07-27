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

from nupic.research.frameworks.continuous_learning.multihead.subdataset import (
    get_datasets,
)


def setup(dataset_name, scenario):
    train_datasets, test_datasets = get_datasets(dataset_name, scenario=scenario)
    train_loaders = [DataLoader(train_dataset, batch_size=256, shuffle=True)
                     for train_dataset in train_datasets]
    test_loaders = [DataLoader(test_dataset, batch_size=1000, shuffle=True)
                    for test_dataset in test_datasets]
    return train_loaders, test_loaders


class MultiHeadTest(unittest.TestCase):

    def test_task_scenario(self):
        """
        Test all target values to be in {0, 1} in the task scenario.
        """
        train_loaders, test_loaders = setup("splitMNIST", "task")
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
        train_loaders, test_loaders = setup("splitMNIST", "domain")
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
        train_loaders, test_loaders = setup("splitMNIST", "class")
        for task_num, train_loader in enumerate(train_loaders):
            min_target, max_target = 2 * task_num, 2 * task_num + 1
            for _data, target in train_loader:
                self.assertLess(target.max().item(), max_target + 1)
                self.assertGreaterEqual(target.min().item(), min_target)
        for task_num, test_loader in enumerate(test_loaders):
            min_target, max_target = 2 * task_num, 2 * task_num + 1
            for _data, target in test_loader:
                self.assertLess(target.max().item(), max_target + 1)
                self.assertGreaterEqual(target.min().item(), min_target)


if __name__ == "__main__":
    unittest.main()
