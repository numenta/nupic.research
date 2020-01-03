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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

TRAIN_DIR = "train"
VAL_DIR = "val"
# TRAIN_DIR = "sz/160/train"
# VAL_DIR = "sz/160/val"

DATA_PATH = Path("~/nta/data/imagenet").expanduser()
TRAIN_PATH = DATA_PATH / TRAIN_DIR
VAL_PATH = DATA_PATH / VAL_DIR
TRAIN_FILES = TRAIN_PATH.glob("*/*.JPEG")
VAL_FILES = VAL_PATH.glob("*/*.JPEG")
HDF5_FILE = DATA_PATH / "imagenet.hdf5"


def resize(sz, image_path):
    with Image.open(image_path) as img:
        # Resize image preserving aspect ratio
        w, h = img.size
        ratio = min(h / sz, w / sz)
        resized_img = img.resize((int(w / ratio), int(h / ratio)),
                                 resample=Image.BICUBIC)
    return resized_img


def hdf5_save(group_name, lock, image_path):
    """
    Save imagenet images to HDF5

    :param group_name: top level group name ("train", "val", etc)
    :param lock: Lock object used to control write access to hdf5 file
    :param image_path: Path object for the image file
    """
    image_name = image_path.name
    wnid = image_path.parent.name
    image_data = image_path.read_bytes()

    lock.acquire()
    hdf5_file = h5py.File(name=HDF5_FILE, mode="a")
    try:
        main_group = hdf5_file.require_group(group_name)
        wnid_group = main_group.require_group(wnid)
        wnid_group.create_dataset(image_name, data=np.void(image_data))
    finally:
        hdf5_file.close()
        lock.release()

    return image_name


def main():
    with ProcessPoolExecutor(multiprocessing.cpu_count()) as executor, \
            multiprocessing.Manager() as manager:

        lock = manager.Lock()
        hdf5_val = partial(hdf5_save, VAL_DIR, lock)
        results = executor.map(hdf5_val, VAL_FILES)
        for _ in tqdm(results, desc="Saving validation dataset"):
            pass

        hdf5_train = partial(hdf5_save, TRAIN_DIR, lock)
        results = executor.map(hdf5_train, TRAIN_FILES)
        for _ in tqdm(results, desc="Saving training dataset"):
            pass


if __name__ == "__main__":
    main()
