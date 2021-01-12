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

from filelock import FileLock
from tqdm import tqdm

from nupic.research.frameworks.pytorch.dataset_utils import PreprocessedDataset

DATA_URL = "http://public.numenta.com/datasets/google_speech_commands/gsc_preprocessed_v0.01.tar.gz"  # noqa: E501


__all__ = [
    "preprocessed_gsc",
    "download_gsc_data",
]


def preprocessed_gsc(root, train=True, qualifiers=None, download=True):
    """
    Create train or test dataset from preprocessed GSC data, downloading if
    necessary.

    Warning: Be sure to call dataset.load_next() following each epoch of training.
    Otherwise, no new augmentations will be loaded, and the same exact samples
    will be reused.

    .. note:: To load the preprocessed noise dataset use noise levels
              `["05", "10", ..., "50"]` as qualifiers on the test dataset.
              Noise level "00" is equivalent to the regular "test" dataset
    .. seealso:: PreprocessedDataset

    :param root: directory to store or load downloaded data
    :param train: whether to load train or test data
    :param qualifiers: List of qualifiers for each preprocessed files in this dataset.
           If None, `range(30)` will be used for training and "00" for testing.
    :param download: whether to download the data
    """

    root = os.path.expanduser(root)
    if download:
        download_gsc_data(root)

    if train:
        basename = "gsc_train"
        if qualifiers is None:
            qualifiers = range(30)
    else:
        # Load test and noise dataset
        basename = "gsc_test_noise"
        if qualifiers is None:
            qualifiers = ["00"]

    dataset = PreprocessedDataset(
        cachefilepath=root,
        basename=basename,
        qualifiers=qualifiers,
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
