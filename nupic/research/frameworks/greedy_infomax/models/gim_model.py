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
from .gim_block import GreedyInfoMaxBlock, InfoEstimateAggregator, EncodingAggregator
from .utility_layers import PatchifyInputs
class GreedyInfoMaxModel(nn.Sequential):
    """
    A model which wraps another model and adds Greedy InfoMax functionality. It does
    this by
    1. adding a PatchifyInputs layer to the beginning of the network
    2. adds GIM blocks to the network and connects them using forward hooks on the model
    2. tracks the info estimates using InfoEstimateAggregator
    3. tracks the encodings using the EncodingAggregator
    """
    def __init__(self,
                 model,
                 named_modules,
                 k_predictions=5,
                 negative_samples=16,
                 patch_size=5,
                 overlap=2):
        pass
        self.training_mode = "unsupervised"
        self.model = model
        self.gim_blocks = []
        self.info_estimate_aggregator = InfoEstimateAggregator(k_predictions,
                                                               negative_samples)
        self.encoding_aggregator = EncodingAggregator()
        self.create_gim_blocks(named_modules, model) # TODO: create this method

        self.patchify_inputs = PatchifyInputs(patch_size, overlap)

    def set_training_mode(self, training_mode="unsupervised"):
        """
        Sets the training mode for the model.
        """
        if training_mode not in ["unsupervised", "supervised"]:
            raise Exception("Training mode must be either 'unsupervised' or 'supervised'")
        self.training_mode = training_mode

        # go through each of the registered modules and set its hook to its
        # corresponding GreedyInfoMaxBlock to the correct mode. For example,
        # in unsupervised training, go through each of the modules and set the hook
        # to the forward_unsupervised function of the GreedyInfoMaxBlock.
        for name, module in self.named_modules():
            if self.training_mode == "unsupervised":
                module.set_forward_hook(module.forward_unsupervised)


    def forward_unsupervised(self, x):
        """
        Forward pass for unsupervised training.
        """
        self.set_mode("unsupervised")
        return self.forward(x)

    def forward_supervised(self, x):
        """
        Forward pass for supervised training.
        """
        pass

