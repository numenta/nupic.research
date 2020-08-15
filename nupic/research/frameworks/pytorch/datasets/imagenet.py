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

import os

import h5py
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop

from nupic.research.frameworks.pytorch.dataset_utils import (
    CachedDatasetFolder,
    HDF5Dataset,
    ImageNetPolicy,
)

IMAGENET_NUM_CLASSES = {
    10: [
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"
    ],
    100: [
        "n01440764", "n01592084", "n01601694", "n01630670", "n01631663",
        "n01664065", "n01677366", "n01693334", "n01734418", "n01751748",
        "n01755581", "n01855672", "n01877812", "n01978287", "n01981276",
        "n02025239", "n02027492", "n02033041", "n02056570", "n02089867",
        "n02091244", "n02091635", "n02093428", "n02094258", "n02104365",
        "n02105251", "n02106662", "n02107312", "n02108422", "n02112350",
        "n02129165", "n02174001", "n02268443", "n02317335", "n02410509",
        "n02423022", "n02454379", "n02457408", "n02488291", "n02497673",
        "n02536864", "n02640242", "n02655020", "n02727426", "n02783161",
        "n02808304", "n02841315", "n02871525", "n02892201", "n02971356",
        "n02979186", "n02981792", "n03018349", "n03125729", "n03133878",
        "n03207941", "n03250847", "n03272010", "n03372029", "n03400231",
        "n03457902", "n03481172", "n03482405", "n03602883", "n03680355",
        "n03697007", "n03763968", "n03791053", "n03804744", "n03837869",
        "n03854065", "n03891332", "n03954731", "n03956157", "n03970156",
        "n03976657", "n04004767", "n04065272", "n04120489", "n04149813",
        "n04192698", "n04200800", "n04252225", "n04259630", "n04332243",
        "n04335435", "n04346328", "n04350905", "n04404412", "n04461696",
        "n04462240", "n04509417", "n04550184", "n04606251", "n07716358",
        "n07718472", "n07836838", "n09428293", "n13040303", "n15075141"
    ],
}


class ImagenetDataset(object):

    def __init__(
        self, data, train_dir="train", val_dir="val", num_classes=1000,
        use_auto_augment=False, sample_transform=None, target_transform=None,
        replicas_per_sample=1
    ):

        self.train_dataset = self.create_train_dataset(
            data_dir=data,
            train_dir=train_dir,
            num_classes=num_classes,
            use_auto_augment=use_auto_augment,
            sample_transform=sample_transform,
            target_transform=target_transform,
            replicas_per_sample=replicas_per_sample,
        )

        self.val_dataset = self.create_validation_dataset(
            data_dir=data,
            val_dir=val_dir,
            num_classes=num_classes,
        )

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return None

    @classmethod
    def create_train_dataset(
        cls, data_dir, train_dir, num_classes=1000, use_auto_augment=False,
        sample_transform=None, target_transform=None, replicas_per_sample=1
    ):
        """
        Configure Imagenet training dataset

        Creates :class:`CachedDatasetFolder` :class:`HDF5Dataset` pre-configured
        for the training cycle

        :param data_dir: The directory or hdf5 file containing the dataset
        :param train_dir: The directory or hdf5 group containing the training data
        :param num_classes: Limit the dataset size to the given number of classes
        :param sample_transform: List of transforms acting on the samples
                                 to be added to the defaults below
        :param target_transform: List of transforms acting on the targets
        :param replicas_per_sample: Number of replicas to create per sample
                                    in the batch (each replica is transformed
                                    independently). Used in maxup.

        :return: CachedDatasetFolder or HDF5Dataset
        """
        if use_auto_augment:
            transform = transforms.Compose(
                transforms=[
                    RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                        inplace=True
                    ),
                ],
            )
        else:
            transform = transforms.Compose(
                transforms=[
                    RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                        inplace=True
                    ),
                ],
            )

        transform = transforms.Compose(
            transforms=[transform] + (sample_transform or []))

        if h5py.is_hdf5(data_dir):
            # Use fixed Imagenet classes if mapping is available
            if num_classes in IMAGENET_NUM_CLASSES:
                classes = IMAGENET_NUM_CLASSES[num_classes]
                dataset = HDF5Dataset(hdf5_file=data_dir, root=train_dir,
                                      classes=classes, transform=transform,
                                      target_transform=target_transform,
                                      replicas_per_sample=replicas_per_sample)
            else:
                dataset = HDF5Dataset(hdf5_file=data_dir, root=train_dir,
                                      num_classes=num_classes, transform=transform,
                                      target_transform=target_transform,
                                      replicas_per_sample=replicas_per_sample)
        else:
            dataset = CachedDatasetFolder(root=os.path.join(data_dir, train_dir),
                                          num_classes=num_classes, transform=transform,
                                          target_transform=target_transform)
        return dataset

    @classmethod
    def create_validation_dataset(cls, data_dir, val_dir, num_classes=1000):
        """
        Configure Imagenet validation dataloader

        Creates :class:`CachedDatasetFolder` or :class:`HDF5Dataset` pre-configured
        for the validation cycle.

        :param data_dir: The directory or hdf5 file containing the dataset
        :param val_dir: The directory containing or hdf5 group the validation data
        :param num_classes: Limit the dataset size to the given number of classes
        :return: CachedDatasetFolder or HDF5Dataset
        """

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    inplace=True
                ),
            ]
        )
        if h5py.is_hdf5(data_dir):
            if num_classes in IMAGENET_NUM_CLASSES:
                classes = IMAGENET_NUM_CLASSES[num_classes]
                dataset = HDF5Dataset(hdf5_file=data_dir, root=val_dir,
                                      classes=classes, transform=transform)
            else:
                dataset = HDF5Dataset(hdf5_file=data_dir, root=val_dir,
                                      num_classes=num_classes, transform=transform)
        else:
            dataset = CachedDatasetFolder(root=os.path.join(data_dir, val_dir),
                                          num_classes=num_classes, transform=transform)
        return dataset
