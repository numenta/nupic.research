#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
import sys

import numpy as np
import torch

from nupic.research.frameworks.pytorch.dataset_utils import PreprocessedDataset


def process_gsc_by_class(data_dir=None):

    if data_dir is not None:
        data_dir = data_dir
    else:
        data_dir = "/home/ec2-user/nta/data/"

    if os.path.isdir(data_dir + "data_classes"):
        pass
    else:
        os.mkdir(data_dir + "data_classes")

    class_min = 1
    class_max = 12

    ranges = [np.arange(k, k + 1) for k in range(30)]

    for k in range(class_min, class_max + 1):
        data_tensor = torch.zeros(0, 1, 32, 32)
        for j in range(len(ranges)):
            dataset = PreprocessedDataset(cachefilepath=data_dir,
                                          basename="gsc_train",
                                          qualifiers=ranges[j])

            class_indices = np.where(dataset.tensors[1] == k)[0]

            if len(class_indices) > 0:
                data_tensor = torch.cat((data_tensor, torch.Tensor(
                    dataset.tensors[0][class_indices, :, :, :])))

        labels_tensor = torch.Tensor(data_tensor.shape[0] * [k - 1]).long()

        out_tensor = list((data_tensor, labels_tensor))
        with open(data_dir + "/data_classes/data_train_{}.npz".format(k), "wb") as f:
            torch.save(out_tensor, f)

    for k in range(class_min, class_max + 1):
        data_tensor = torch.zeros(0, 1, 32, 32)
        dataset = PreprocessedDataset(cachefilepath=data_dir,
                                      basename="gsc_valid",
                                      qualifiers=[""]
                                      )

        class_indices = np.where(dataset.tensors[1] == k)[0]

        if len(class_indices) > 0:
            data_tensor = torch.cat((data_tensor, torch.Tensor(
                dataset.tensors[0][class_indices, :, :, :])))

        labels_tensor = torch.Tensor(data_tensor.shape[0] * [k - 1]).long()

        out_tensor = list((data_tensor, labels_tensor))
        with open(data_dir + "/data_classes/data_valid.npz", "wb") as f:
            torch.save(out_tensor, f)

    noise_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for k in range(class_min, class_max + 1):
        for n in noise_values:
            data_tensor = torch.zeros(0, 1, 32, 32)
            dataset = PreprocessedDataset(cachefilepath=data_dir,
                                          basename="gsc_test_noise",
                                          qualifiers=["{:02d}".format(int(100 * n))])

            class_indices = np.where(dataset.tensors[1] == k)[0]

            if len(class_indices) > 0:
                data_tensor = torch.cat((data_tensor, torch.Tensor(
                    dataset.tensors[0][class_indices, :, :, :])))

            if k == 1:
                labels_tensor = torch.Tensor(55860 * [k - 1]).long()
            else:
                labels_tensor = torch.Tensor(data_tensor.shape[0] * [k - 1]).long()

            out_tensor = list((data_tensor, labels_tensor))
            if n == 0.0:
                tensor_string = "data_test_0noise{}.npz".format(k)
            else:
                tensor_string = "data_test_{}_{}.npz".format(k,n)

            with open(data_dir + "/data_classes/" + tensor_string, "wb") as f:
                torch.save(out_tensor, f)


if __name__ == "__main__":
    process_gsc_by_class()
