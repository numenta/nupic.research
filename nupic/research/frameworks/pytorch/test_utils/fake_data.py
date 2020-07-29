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

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

__all__ = [
    "FakeDataPredefinedTargets",
    "FakeDataLoader",
]


class FakeDataPredefinedTargets(FakeData):
    """
    A fake data class similar to `FakeData`, but one that has a `targets` attribute,
    specified upon initialization instead of generating a new target variable each time
    a sample is drawn.
    """

    def __init__(self, size=1000, image_size=(3, 224, 224), num_classes=10):
        super().__init__(size=size, image_size=image_size, num_classes=num_classes,
                         transform=transforms.ToTensor())
        self.targets = torch.randint(0, self.num_classes, size=(size,),
                                     dtype=torch.long)

    def __getitem__(self, index):
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = self.targets[index].item()
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


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
            transform = transforms.ToTensor()
        self.fake_dataset = FakeData(
            size=dataset_size, image_size=image_size, num_classes=num_classes,
            transform=transform, target_transform=target_transform,
            random_offset=random_offset,
        )

        super().__init__(self.fake_dataset, batch_size=batch_size)
