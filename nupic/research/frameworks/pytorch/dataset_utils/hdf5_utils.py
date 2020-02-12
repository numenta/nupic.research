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

import io

import h5py
import numpy as np
import torch
from torchvision.transforms import ToPILImage

__all__ = [
    "tensor_to_byte_array",
    "HDF5DataSaver",
]


def tensor_to_byte_array(tensor):
    tensor = tensor.detach().cpu()
    byte_io = io.BytesIO()
    torch.save(tensor, byte_io)
    img_bytes = byte_io.getvalue()
    return np.void(img_bytes)


def tensor_to_image_to_byte_array(tensor):
    tensor = tensor.detach().cpu()
    image = ToPILImage(mode=None)(tensor)
    byte_io = io.BytesIO()
    image.save(byte_io, format="PNG")
    img_bytes = byte_io.getvalue()
    return np.void(img_bytes)


class HDF5DataSaver(object):

    def __init__(self, data_path, lock=None, to_bytes_func=None):

        self.data_path = data_path
        self.lock = lock
        self.to_bytes_func = to_bytes_func or tensor_to_image_to_byte_array

    @staticmethod
    def hdf5_save(
        data_path, image_data, group_name, class_name, image_name, lock=None
    ):
        """
        Save imagenet images to HDF5

        :param group_name: top level group name ("train", "val", etc)
        :param lock: Lock object used to control write access to hdf5 file
        :param image_path: Path object for the image file
        """

        wnid = class_name
        if lock is not None:
            # print(" " * 10, "before lock acquire")
            lock.acquire()
            # print(" " * 10, "after lock acquire")
        hdf5_file = h5py.File(name=data_path, mode="a")
        try:
            main_group = hdf5_file.require_group(group_name)
            wnid_group = main_group.require_group(wnid)
            wnid_group.create_dataset(image_name, data=image_data)
        finally:
            hdf5_file.close()
            if lock is not None:
                # print(" " * 10, "before lock release")
                lock.release()
                # print(" " * 10, "after lock release")

        return image_name

    def append_tensor(self, tensor, image_name, group_name, class_name):

        image_data = self.to_bytes_func(tensor)
        data_path = self.data_path
        lock = self.lock
        self.hdf5_save(
            data_path, image_data, group_name, class_name, image_name, lock)
