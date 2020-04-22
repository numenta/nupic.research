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

import os
import re
from pathlib import Path

import numpy as np
import torch


def dataset_from_npz(filepath):
    x, y = np.load(filepath).values()
    x, y = map(torch.tensor, (x, y))
    return torch.utils.data.TensorDataset(x, y)


class PreprocessedGSC(object):
    def __init__(self):
        self.folder = Path(
            os.path.expanduser("~/nta/data/gsc/gsc_preprocessed")
        )
        matches = [re.search(r"gsc_train(\d+).npz", filename)
                   for filename in os.listdir(self.folder)]
        self.seeds = [int(match.group(1))
                      for match in matches
                      if match is not None]

    def get_train_dataset(self, iteration):
        seed = self.seeds[iteration % len(self.seeds)]
        filename = "gsc_train{}.npz".format(seed)
        return dataset_from_npz(self.folder / filename)

    def get_validation_dataset(self):
        filename = "gsc_valid.npz"
        return dataset_from_npz(self.folder / filename)

    def get_test_dataset(self, noise_level=0.0):
        filename = "gsc_test_noise{}.npz".format("{:.2f}".format(noise_level)[2:])
        return dataset_from_npz(self.folder / filename)
