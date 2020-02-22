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

from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

__all__ = [
    "FakeDataLoader",
]


class FakeDataLoader(DataLoader):
    """
    This class wraps the dataset `FakeData` to reduce the overhead of incorporating
    the fake dataset into a `DataLoader` and turning the images to `torch.Tensors`
    upon readout.
    """

    def __init__(
        self,
        batch_size=2, dataset_size=10, image_size=(3, 224, 224), num_classes=10,
        transform=None, target_transform=None, random_offset=0,
    ):
        if transform is None:
            transform = ToTensor()
        self.fake_dataset = FakeData(
            size=dataset_size, image_size=image_size, num_classes=num_classes,
            transform=transform, target_transform=target_transform,
            random_offset=random_offset,
        )

        super().__init__(self.fake_dataset, batch_size=batch_size)
