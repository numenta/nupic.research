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

import numpy as np
import torch
from torchvision import datasets


class MNISTSplitter(object):
    """ Just a simple class for downloading and splitting your MNIST dataset
        :param data_dir: Where you want the data to live. Defaults to None
    """

    def __init__(self, data_dir=None):

        if data_dir is None:
            data_dir = "/home/ec2-user/nta/data/mnist/"

        if not os.path.isdir(data_dir):
            print("Making directory {}".format(data_dir))
            os.mkdir(data_dir)

        if len(os.listdir(data_dir)) > 0.0:
            print("Warning: will delete and replace local files")
            for file_path in os.listdir(data_dir):
                try:
                    os.remove(os.path.join(data_dir, file_path))
                except OSError as err:
                    print("Error {} : {}".format(os.path.join(data_dir,
                                                              file_path), err))

        self.data_dir = data_dir
        self.num_classes = 10

        self.train_data, self.test_data = self.get_datasets(self.data_dir)

    def get_datasets(self, data_dir):
        """ Get the datasets
        """
        train_dataset = datasets.MNIST(data_dir, download=True, train=True)

        test_dataset = datasets.MNIST(data_dir, download=True, train=False)

        print("Saved data to {}".format(data_dir))

        return train_dataset, test_dataset

    def split_mnist(self, data_dir):
        """ Get tensors for each class and save them individually
        """
        xs_train, ys_train = self.train_data.data, self.train_data.targets
        xs_test, ys_test = self.test_data.data, self.test_data.targets

        for class_ in range(self.num_classes):
            # Training data
            y_inds = np.where(ys_train == class_)[0]
            x_class = xs_train[y_inds, :, :]
            torch.save(
                (x_class, class_ * torch.ones(len(y_inds))),
                data_dir + "/mnist_train_{}.npz".format(class_),
            )

            # Test data
            y_inds = np.where(ys_test == class_)[0]
            x_class = xs_test[y_inds, :, :]
            torch.save(
                (x_class, class_ * torch.ones(len(y_inds))),
                data_dir + "/mnist_test_{}.npz".format(class_),
            )


if __name__ == "__main__":

    data_dir = "/home/ec2-user/nta/data/mnist/"
    splitter = MNISTSplitter(data_dir=data_dir)
    print("Splitting... ")
    splitter.split_mnist(data_dir)
    print("Done!")
