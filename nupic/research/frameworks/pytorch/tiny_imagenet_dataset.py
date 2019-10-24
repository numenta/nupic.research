# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from __future__ import division, print_function

import os
import shutil

import pandas as pd
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

ARCHIVE_DICT = {
    "url": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "md5": "90528d7ca1a48142e341f4ef8d21d0de",
}
VAL_ANNOTATIONS = "val_annotations.txt"
META_FILE = "words.txt"
DATASET_FOLDER = "tiny-imagenet-200"


class TinyImageNet(ImageFolder):
    """`Tiny ImageNet <https://tiny-imagenet.herokuapp.com/>`Classification Dataset.
    Based on ImageNet <http://www.image-net.org/challenges/LSVRC/2014/>

    Args:
        root (string): Root directory of the TinyImageNet Dataset.
        train (boolean, optional): If true, loads training set, otherwise validation set
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, train=True, download=False, **kwargs):

        # load different models whether training or validation
        self.train = train
        img_folder = os.path.join(os.path.expanduser(root), DATASET_FOLDER)
        self.meta_file = os.path.join(img_folder, META_FILE)
        if self.train:
            img_folder = os.path.join(img_folder, "train")
        else:
            img_folder = os.path.join(img_folder, "val")

        # if for the first time, download
        if download:
            self.download(root, img_folder)

        super(TinyImageNet, self).__init__(img_folder, **kwargs)

        # extra attributes for easy reference
        wnid_to_classes = self._load_meta_file()
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }

    def download(self, root, img_folder):

        # regular download
        if not check_integrity(self.meta_file):
            download_and_extract_archive(
                ARCHIVE_DICT["url"], root, md5=ARCHIVE_DICT["md5"]
            )
        else:
            print("Dataset already downloaded.")

        # if validation, take extra step to organize images if not yet
        if not self.train and os.path.isdir(os.path.join(img_folder, "images")):
            print("Rearranging validation folder.")
            annotations = self._load_val_annotations(img_folder)
            prepare_val_folder(img_folder, annotations)
        else:
            print("Validation set rearranged")

    def _load_meta_file(self):
        # TODO: make it faster
        mapping = pd.read_csv(self.meta_file, sep="\t", index_col=None, header=None)
        return {wnid: classes for _, (wnid, classes) in mapping.iterrows()}

    def _load_val_annotations(self, img_folder):
        annotations_file = os.path.join(img_folder, VAL_ANNOTATIONS)
        return pd.read_csv(annotations_file, sep="\t", index_col=None, header=None)


def prepare_val_folder(img_folder, annotations):
    # create folders
    for wnid in annotations.iloc[:, 1].unique():
        os.mkdir(os.path.join(img_folder, wnid))
    # move files
    for _, (img_file, wnid) in annotations.iloc[:, :2].iterrows():
        img_path = os.path.join(img_folder, "images", img_file)
        shutil.move(
            img_path, os.path.join(img_folder, wnid, os.path.basename(img_file))
        )
    # delete images file
    os.rmdir(os.path.join(img_folder, "images"))
