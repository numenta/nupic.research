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
import tempfile

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from nupic.research.frameworks.pytorch.dataset_utils import PreprocessedDataset


def mnist_classwise_loader(
    data_dir="/home/ec2-user/nta/data/mnist",
    transform=None,
    batch_size=32,
):
    """ Create DataLoader instances for the MNIST train and test sets.
        :param data_dir: Where your MNIST data lives.
        :param transform: Any transforms you'd like to apply to your data.
         Defaults to None.
        :param batch_size: Desired batch_size
    """
    std_transform = transforms.Lambda(lambda x: (x[0].float(), x[1].long()))
    if transform is not None:
        transform = transforms.Compose([
            std_transform,
            transform,
        ])

    else:
        transform = std_transform

    train_loader, test_loader = [], []
    num_classes = 10

    for class_ in range(num_classes):
        train_dataset = ClasswiseDataset(
            cachefilepath=data_dir,
            basename="mnist_train_",
            qualifiers=[class_],
            transform=transform,
        )

        train_loader.append(
            DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
            )
        )

        test_dataset = ClasswiseDataset(
            cachefilepath=data_dir,
            basename="mnist_test_",
            qualifiers=[class_],
            transform=transform,
        )

        test_loader.append(
            DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
            )
        )

    return train_loader, test_loader


def combine_classes(
        data_dir,
        training_classes,
        batch_size=32):
    """ Create a DataLoader instance with the classes of your choice,
        for sequential learning.
        :param data_dir: Where your data live.
        :param training_classes: the classes you want to combine
        :type training_classes: iterable (tuple or list)
    """
    data = []
    for k in training_classes:
        data.append(torch.load(data_dir + "mnist_{}.npz".format(k)))

    samples_ = [data[k][0] for k in range(len(training_classes))]
    labels_ = [data[k][1] for k in range(len(training_classes))]
    combined_samples = torch.cat(samples_)
    combined_labels = torch.cat(labels_)
    combined_dataset = list((combined_samples, combined_labels))

    f = tempfile.NamedTemporaryFile(delete=True)
    torch.save(combined_dataset, f)
    dataset = ClasswiseDataset(
        cachefilepath=os.path.split(f.name)[0],
        basename=os.path.split(f.name)[1],
        qualifiers=["tmp"],
        transform=transforms.Compose(
            [transforms.Lambda(lambda x: (x[0].float(), x[1].long()))]
        ),
    )

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    f.flush()

    train_loader = data_loader
    del samples_, labels_, data
    return train_loader
    f.close()


class ClasswiseDataset(PreprocessedDataset):
    """ Identical to PreprocessedDataset except
        it allows for temporary files
        (created when combining classes) and
        uses torch.load instead of numpy.load
        (to avoid use_pickle errors)
    """

    def load_qualifier(self, qualifier):
        """
        Call this to load the a copy of a dataset with the specific qualifier into
        memory.

        :return: Name of the file that was actually loaded.
        """
        if qualifier == "tmp":  # allowance for temp. files
            file_name = os.path.join(self.path, self.basename)
        else:
            file_name = os.path.join(
                self.path, self.basename + "{}.npz".format(qualifier)
            )
        self.tensors = list(torch.load(file_name))

        return file_name
