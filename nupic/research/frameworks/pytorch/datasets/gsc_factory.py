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
import tarfile
import urllib

import numpy as np
import torch.utils.data
from filelock import FileLock
from tqdm import tqdm

from nupic.research.frameworks.pytorch.dataset_utils import (
    FunctionalPreprocessedDataset,
)

DATA_URL = "http://public.numenta.com/datasets/google_speech_commands/gsc_preprocessed_v0.01.tar.gz"  # noqa: E501


__all__ = [
    "preprocessed_gsc",
    "download_gsc_data",
]


def preprocessed_gsc(root, train=True, download=True):
    """
    Create train or validation dataset from preprocessed GSC data, downloading if
    necessary.

    Warning: During training, be sure to call dataset.get_variant(), and call it
    with different indices to train with different augmentations.

    :param root: directory to store or load downloaded data
    :param train: whether to load train of validation data
    :param download: whether to download the data
    """

    root = os.path.expanduser(root)
    if download:
        download_gsc_data(root)

    if train:
        dataset = FunctionalPreprocessedDataset(
            cachefilepath=root,
            basename="gsc_train",
            qualifiers=range(30),
        )
    else:
        x, y = np.load(os.path.join(root, "gsc_valid.npz")).values()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x), torch.tensor(y)
        )

    return dataset


def download_gsc_data(path):
    """
    Download preprocessed GSC dataset
    """
    if os.path.isdir(path):
        return

    os.makedirs(path, exist_ok=True)
    root = os.path.expanduser(path)

    print("Downloading Preprocessed GSC:", DATA_URL)
    with FileLock(f"{root}.lock"):

        with urllib.request.urlopen(DATA_URL) as stream, tarfile.open(
            fileobj=stream, mode="r|*"
        ) as tar, tqdm(total=stream.length) as progress:

            for member in tar:
                progress.set_description(member.name)
                size = stream.length
                tar.extract(path=path, member=member)
                progress.update(size - stream.length)
