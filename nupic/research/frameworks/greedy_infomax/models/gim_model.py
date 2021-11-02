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
# This work was built off the Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The original Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gim_block import GreedyInfoMaxBlock

class GreedyInfoMaxModel(nn.Sequential):
    """
    A model which wraps another model and adds Greedy InfoMax functionality. It does
    this by adding a PatchifyInputs layer to the beginning of the network, and also
    tracks the info estimates/encodings with the InfoEstimateAggregator and
    EncodingAggregator respectively.

    """
    def __init__(self,
                 model,
                 named_modules,
                 k_predictions=5,
                 negative_samples=16,
                 patch_size=5,
                 overlap=2):
        pass

    def set_training_mode(self, training_mode="unsupervised"):
        if training_mode not in ["unsupervised", "supervised"]:
            raise Exception("Training mode must be either 'unsupervised' or 'supervised'")
        pass

