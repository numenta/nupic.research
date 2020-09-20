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

import torch

from nupic.research.frameworks.continual_learning.multihead.multihead import (
    active_class_outputs,
    get_active_classes,
    get_target_transform,
)
from nupic.research.frameworks.pytorch.test_utils.fake_data import FakeDataLoader
from nupic.torch.modules import Flatten


def setup_dataloaders():
    train_loaders = [FakeDataLoader(batch_size=256, image_size=(1, 28, 28))
                     for n in range(5)]
    test_loaders = [FakeDataLoader(batch_size=256, image_size=(1, 28, 28))
                    for n in range(5)]
    return train_loaders, test_loaders


def setup_model(in_features, out_features):
    return torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(in_features, out_features)
    )


class MultiHeadTest(unittest.TestCase):

    def test_active_class_outputs_task(self):
        """
        Test the multihead output size for the task scenario.
        """
        model = setup_model(784, 10)
        train_loaders, test_loaders = setup_dataloaders()
        for data, _target in train_loaders[1]:
            output = active_class_outputs(model, data, [2, 3])
            self.assertEqual(output.shape[1], 2)
        for data, _target in test_loaders[1]:
            output = active_class_outputs(model, data, [2, 3])
            self.assertEqual(output.shape[1], 2)

    def test_active_class_outputs_domain(self):
        """
        Test the multihead output size for the domain scenario.
        """
        model = setup_model(784, 2)
        train_loaders, test_loaders = setup_dataloaders()
        for data, _target in train_loaders[1]:
            output = active_class_outputs(model, data, None)
            self.assertEqual(output.shape[1], 2)
        for data, _target in test_loaders[1]:
            output = active_class_outputs(model, data, None)
            self.assertEqual(output.shape[1], 2)

    def test_active_class_outputs_class(self):
        """
        Test the multihead output size for the class scenario.
        """
        model = setup_model(784, 10)
        train_loaders, test_loaders = setup_dataloaders()
        for data, _target in train_loaders[2]:
            output = active_class_outputs(model, data, [0, 1, 2, 3, 4, 5])
            self.assertEqual(output.shape[1], 6)
        for data, _target in test_loaders[2]:
            output = active_class_outputs(model, data, [0, 1, 2, 3, 4, 5])
            self.assertEqual(output.shape[1], 6)

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

    def test_get_target_transform_task(self):
        """
        Test the target transform to ensure it is correctly transforming dataset target
        variables for the task scenario.
        """
        target_transform = get_target_transform("task")
        targets_task_1 = torch.randint(0, 2, (10,))
        targets_task_2 = torch.randint(2, 4, (10,))
        if target_transform is not None:
            targets_task_1 = target_transform(targets_task_1)
            targets_task_2 = target_transform(targets_task_2)
        self.assertLess(targets_task_1.max().item(), 2)
        self.assertGreaterEqual(targets_task_1.min().item(), 0)
        self.assertLess(targets_task_2.max().item(), 2)
        self.assertGreaterEqual(targets_task_2.min().item(), 0)

    def test_get_target_transform_domain(self):
        """
        Test the target transform to ensure it is correctly transforming dataset target
        variables for the domain scenario.
        """
        target_transform = get_target_transform("domain")
        targets_task_1 = torch.randint(0, 2, (10,))
        targets_task_2 = torch.randint(2, 4, (10,))
        if target_transform is not None:
            targets_task_1 = target_transform(targets_task_1)
            targets_task_2 = target_transform(targets_task_2)
        self.assertLess(targets_task_1.max().item(), 2)
        self.assertGreaterEqual(targets_task_1.min().item(), 0)
        self.assertLess(targets_task_2.max().item(), 2)
        self.assertGreaterEqual(targets_task_2.min().item(), 0)

    def test_get_target_transform_class(self):
        """
        Test the target transform to ensure it is correctly transforming dataset target
        variables for the class scenario.
        """
        target_transform = get_target_transform("class")
        targets_task_1 = torch.randint(0, 2, (10,))
        targets_task_2 = torch.randint(2, 4, (10,))
        if target_transform is not None:
            targets_task_1 = target_transform(targets_task_1)
            targets_task_2 = target_transform(targets_task_2)
        self.assertLess(targets_task_1.max().item(), 2)
        self.assertGreaterEqual(targets_task_1.min().item(), 0)
        self.assertLess(targets_task_2.max().item(), 4)
        self.assertGreaterEqual(targets_task_2.min().item(), 2)


if __name__ == "__main__":
    unittest.main()
