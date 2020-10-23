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

from os.path import isdir, isfile, join

from torchvision.datasets import Omniglot
from torchvision.datasets.utils import extract_archive

__all__ = ["OmniglotFromZip"]


class OmniglotFromZip(Omniglot):

    def download(self):
        """
        Try and extract Omniglot from a zipfile before downloading the dataset. If the
        dataset is already downloaded and extracted, this function will do nothing.

        Make sure `download=True` to enable extraction.
        """

        # Target folder is either "images_background" or "images_evaluation"
        target_folder = self._get_target_folder()
        zip_filename = target_folder + ".zip"
        zip_path = join(self.root, zip_filename)
        extract_folder = join(self.root, target_folder)

        # Extract zip file if it exists and hasn't already been extracted.
        if isfile(zip_path) and not isdir(extract_folder):
            print("Extracting {} to {}".format(zip_path, self.root))
            extract_archive(
                from_path=zip_path,
                to_path=self.root,
                remove_finished=False,  # don't delete the zip file
            )

        super().download()
