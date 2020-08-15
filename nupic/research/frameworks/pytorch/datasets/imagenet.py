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

from nupic.research.frameworks.pytorch.imagenet.experiment_utils import (
    create_train_dataset,
    create_validation_dataset,
)


class ImagenetDataset(object):

    def __init__(
        self, data, train_dir="train", val_dir="val", num_classes=1000,
        use_auto_augment=False, sample_transform=None, target_transform=None,
        replicas_per_sample=1
    ):

        self.train_dataset = create_train_dataset(
            data_dir=data,
            train_dir=train_dir,
            num_classes=num_classes,
            use_auto_augment=use_auto_augment,
            sample_transform=sample_transform,
            target_transform=target_transform,
            replicas_per_sample=replicas_per_sample,
        )

        self.val_dataset = create_validation_dataset(
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
