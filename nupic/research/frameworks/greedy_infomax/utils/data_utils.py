# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
#
# This work was based on the original Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------

from torchvision import transforms


"""
Composes transforms specific to GreedyInfoMax experiments according to an arguments
dictionary.
:param val: An optional boolean which specifies whether the dataset is for a training or
            validation. Defaults to False.
:param aug: A dictionary which specifies the augmentations to apply to the dataset.
            Keys include "randcrop" and "flip", which determine whether to randomly
            crop or flip the samples respectively.

"""


def get_transforms(val=False, aug=None):
    trans = []
    if aug["randcrop"]:
        if not val:
            trans.append(transforms.RandomCrop(aug["randcrop"]))
        else:
            trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not val:
        trans.append(transforms.RandomHorizontalFlip())

    trans.append(transforms.Grayscale())
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))

    trans = transforms.Compose(trans)
    return trans
