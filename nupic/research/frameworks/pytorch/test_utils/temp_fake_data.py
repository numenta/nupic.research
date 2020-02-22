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

import os
import tempfile

from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.pytorch.dataset_utils import HDF5DataSaver
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import (
    create_train_dataloader,
    create_validation_dataloader,
)

__all__ = [
    "TempFakeSavedData",
]


class TempFakeSavedData(object):
    """
    This class:
       1) Saves data in hdf5 format to a temporary directory managed by `tempfile`.
       2) Instantiates `DataLoader`s and `Datasets`'s' to iterate over the saved data.
       3) Deletes the saved data upon `cleanup()` or `__exit__`.

    Example:
    ```
    temp_data = TempFakeSavedData()
    train_loader = temp_data.train_dataloader

    batches = 0
    for image, target in train_loader:
        assert image.shape == (3, 224, 224)
        batch += 1
    assert batches == 5

    temp_data.cleanup()
    ```
    """

    def __init__(
        self,
        dataset_name=None,
        batch_size=2, train_size=10, val_size=10,
        image_size=(3, 224, 224), num_classes=10,
        transform=None, target_transform=None, random_offset=0, num_workers=0
    ):

        self.dataset_name = dataset_name or "temp_data.hdf5"
        if ".hdf5" not in self.dataset_name[-5:]:
            self.dataset_name = self.dataset_name + ".hdf5"

        # Create Fake directory of data.
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create saver to help store new data.
        self.dataset_path = os.path.join(self.temp_dir.name, self.dataset_name)
        self.data_saver = HDF5DataSaver(data_path=self.dataset_path)

        # Create fake datasets that help generate data.
        self.fake_train_dataset = FakeData(
            size=train_size, image_size=image_size, num_classes=num_classes,
            transform=transform, target_transform=target_transform,
            random_offset=random_offset
        )
        val_random_offset = random_offset + max(int(num_classes / 2), 1)
        self.fake_val_dataset = FakeData(
            size=val_size, image_size=image_size, num_classes=num_classes,
            transform=transform, target_transform=target_transform,
            random_offset=val_random_offset
        )

        # Keep track of classes; small sampling sizes may give incomplete coverage.
        # e.g. `train_size=10` may yield sampled classes, say, 1-6 and not all 10.
        self.classes = {"train": set(), "val": set()}

        # Append one pass of each dataset to the "train" and "val" hdf5 groupings.
        self._append_dataset(
            dataset=self.fake_train_dataset, group_name="train")
        self._append_dataset(
            dataset=self.fake_val_dataset, group_name="val")

        # Create dataloaders of newly saved temp data.
        self.batch_size = batch_size
        self.train_num_classes = len(self.classes["train"])
        self.val_num_classes = len(self.classes["val"])
        self.train_dataloader = create_train_dataloader(
            self.dataset_path, "train", self.batch_size,
            workers=num_workers, num_classes=self.train_num_classes, distributed=False
        )
        self.val_dataloader = create_validation_dataloader(
            self.dataset_path, "val", batch_size,
            workers=num_workers, num_classes=self.val_num_classes
        )

        # Set the datasets that correspond to the dataloaders.
        self.train_dataset = self.train_dataloader.dataset
        self.val_dataset = self.val_dataloader.dataset

    def _append_dataset(self, dataset, group_name):

        transform = ToTensor()
        classes = []
        for i, (image, target) in enumerate(dataset):

            image_tensor = transform(image)
            image_name = "img_" + str(i) + ".png"
            class_name = "class_" + str(target.item())
            classes.append(target.item())

            self.data_saver.append_tensor(
                image_tensor, image_name, group_name, class_name)

        self.classes[group_name].update(set(classes))

    @property
    def data_dir(self):
        return self.temp_dir.name

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        self.temp_dir.cleanup()

    def cleanup(self):
        self.temp_dir.cleanup()
