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
  """
  Add noise to random pixels in images
  """


  def __init__(self,
               noiselevel=0.0,
               highValue=0.1307 + 2 * 0.3081,
               lowValue=0.1307 + 2*0.3081,
               logDir=None, logProbability=0.01):
    """
    An image transform that adds noise to random elements in the image array.
    Half the time the noise value is highValue and the other half it is
    lowValue (by default highValue and lowValue are the same). Suggested values
    are 'mean +/- 2*stdev'

    :param noiselevel:
      From 0 to 1. For each pixel, set its value to a noise value with this
      probability.

    :param logDir:
      If set to a directory name, then save a random sample of the images to
      this directory.

    :param logProbability:
      The percentage of samples to save to the log directory.

    """
    self.noiseLevel = noiselevel
    self.highValue = highValue
    self.lowValue = lowValue
    self.iteration = 0
    self.logDir = logDir
    self.logProbability = logProbability


  def __call__(self, image):
    self.iteration += 1
    if self.noiseLevel > 0.0:
      a = image.view(-1)
      numNoiseBits = int(a.shape[0] * self.noiseLevel)
      permutedIndices = np.random.permutation(a.shape[0])
      a[permutedIndices[0:numNoiseBits // 2]] = self.highValue
      a[permutedIndices[numNoiseBits // 2:numNoiseBits]] = self.lowValue

    # Save a subset of the images for debugging
    if self.logDir is not None:
      if np.random.random() <= self.logProbability:
        outfile = os.path.join(self.logDir,
                               "im_noise_" + str(int(self.noiseLevel * 100)) + "_"
                               + str(self.iteration).rjust(6, '0') + ".png")
        save_image(image, outfile)

    return image
