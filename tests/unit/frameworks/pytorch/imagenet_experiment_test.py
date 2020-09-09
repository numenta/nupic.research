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

import os
import unittest

import torch

from nupic.research.frameworks.vernon import ImagenetExperiment
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.research.frameworks.pytorch.test_utils import TempFakeSavedData


class ImagenetExperimentTest(unittest.TestCase):

    def setUp(self):
        self.config = dict(

            num_classes=10,

            # Dataset path
            data=os.path.expanduser("~"),

            # Number of epochs
            epochs=1,

            # Model class. Must inherit from "torch.nn.Module"
            model_class=resnet50,
            # model model class arguments passed to the constructor
            model_args=dict(),

            # Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.SGD,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(
                lr=0.1,
                weight_decay=1e-04,
                momentum=0.9,
                dampening=0,
                nesterov=True
            ),

            # Learning rate scheduler class. Must inherit from "_LRScheduler"
            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            # Learning rate scheduler class class arguments passed to the constructor
            lr_scheduler_args=dict(
                # LR decayed by 10 every 30 epochs
                gamma=0.1,
                step_size=30,
            ),

            # Works with progressive_resize and the available GPU memory fitting as many
            # images as possible in each batch - dict(start_epoch: batch_size)
            dynamic_batch_size=None,

            # Loss function. See "torch.nn.functional"
            loss_function=torch.nn.functional.cross_entropy,

        )

    def test_temp_data_util(self):

        temp_data = TempFakeSavedData(
            train_size=10,
            batch_size=2,
        )

        temp_data_path = temp_data.dataset_path
        with temp_data as data:
            i = 0
            for image, _ in data.train_dataloader:
                batch_size = image.shape[0]
                self.assertEqual(batch_size, 2)
                i += 1
            self.assertEqual(i, 5)  # there should be 10 / 2 batches

        # Validate data was deleted.
        data_exists = os.path.exists(temp_data_path)
        self.assertFalse(data_exists)

    def test_init_with_fake_data(self):

        exp = ImagenetExperiment()
        temp_data = TempFakeSavedData(
            train_size=12,
            batch_size=2,
            num_classes=10,
        )

        temp_data_path = temp_data.dataset_path
        with temp_data as data:

            self.config["data"] = data.dataset_path
            self.config["num_classes"] = data.train_num_classes
            self.config["batch_size"] = 4
            exp.setup_experiment(self.config)

            i = 0
            for image, _ in exp.train_loader:
                batch_size = image.shape[0]
                self.assertEqual(batch_size, 4)
                i += 1
            self.assertEqual(i, 3)  # there should be 12 / 4 batches

        # Validate data was deleted.
        data_exists = os.path.exists(temp_data_path)
        self.assertFalse(data_exists)


if __name__ == "__main__":
    unittest.main(verbosity=2)
