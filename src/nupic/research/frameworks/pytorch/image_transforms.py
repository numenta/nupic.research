# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
from torchvision.utils import save_image


class RandomNoise(object):
    """Add noise to random pixels in images."""

    def __init__(
        self,
        noise_level=0.0,
        high_value=0.1307 + 2 * 0.3081,
        low_value=0.1307 + 2 * 0.3081,
        log_dir=None,
        log_probability=0.01,
    ):
        """An image transform that adds noise to random elements in the image
        array. Half the time the noise value is high_value and the other half
        it is low_value (by default high_value and low_value are the same).
        Suggested values are 'mean +/- 2*stdev'.

        :param noise_level:
          From 0 to 1. For each pixel, set its value to a noise value with this
          probability.

        :param log_dir:
          If set to a directory name, then save a random sample of the images to
          this directory.

        :param log_probability:
          The percentage of samples to save to the log directory.
        """
        self.noise_level = noise_level
        self.high_value = high_value
        self.low_value = low_value
        self.iteration = 0
        self.log_dir = log_dir
        self.log_probability = log_probability

    def __call__(self, image):
        self.iteration += 1
        if self.noise_level > 0.0:
            a = image.view(-1)
            num_noise_bits = int(a.shape[0] * self.noise_level)
            permuted_indices = np.random.permutation(a.shape[0])
            a[permuted_indices[0 : num_noise_bits // 2]] = self.high_value
            a[permuted_indices[num_noise_bits // 2 : num_noise_bits]] = self.low_value

        # Save a subset of the images for debugging
        if self.log_dir is not None:
            if np.random.random() <= self.log_probability:
                outfile = os.path.join(
                    self.log_dir,
                    "im_noise_"
                    + str(int(self.noise_level * 100))
                    + "_"
                    + str(self.iteration).rjust(6, "0")
                    + ".png",
                )
                save_image(image, outfile)

        return image
