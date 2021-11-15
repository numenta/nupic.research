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
    3. tracks the info estimates using InfoEstimateAggregator
    4. tracks the encodings using the EncodingAggregator


    Parameters
    ----------
    model: nn.Module
        The model to wrap
    named_modules: a list of (str, nn.Module) tuples
        The modules to which forward hooks to GIM blocks should be added
    k_predictions: int
        The number of prediction steps
    negative_samples: int
        The number of negative samples to use for contrastive training
    patch_size: int
        The size of the patches to use for the GIM blocks
    overlap: int
        The number of overlapping pixels between patches
    n_patches_x: int
        The number of patches in the x direction
    n_patches_y: int
        The number of patches in the y direction
    """
    def __init__(self,
                 model,
                 modules_and_output_sizes,
                 k_predictions=5,
                 negative_samples=16,
                 patch_size=16,
                 overlap=2,
                 n_patches_x=None,
                 n_patches_y=None,):
        super(GreedyInfoMaxModel, self).__init__()
        self.training_mode = "unsupervised"
        self.gim_blocks = {}  # dict of (module, gim_block) pairs
        self.patchify_inputs = PatchifyInputs(patch_size, overlap)
        self.model = model
        self.info_estimate_aggregator = InfoEstimateAggregator()
        self.encoding_aggregator = EncodingAggregator()
        self.create_gim_blocks(modules_and_output_sizes,
                               k_predictions,
                               negative_samples,
                               n_patches_x, n_patches_y)


    def forward(self, x):
        """
        Forward pass for unsupervised training. Attaches the GIM blocks to the model
        using forward hooks, then collects the info estimates from the info estimate
        aggregator
        """
        for (module, gim_block) in self.gim_blocks.items():
            module._forward_hooks.clear()
            module.register_forward_hook(gim_block.wrapped_forward)
        self.info_estimate_aggregator.clear_outputs()
        super().forward(x)
        return self.info_estimate_aggregator.get_outputs()

    def encode(self, x):
        """
        Forward pass for supervised training.
        """
        for (module, gim_block) in self.gim_blocks.items():
            module._forward_hooks.clear()
            module.register_forward_hook(gim_block.wrapped_encode)
        self.encoding_aggregator.clear_outputs()
        super().forward(x)
        return self.encoding_aggregator.get_outputs()

    def create_gim_blocks(self,
                          modules_and_output_sizes,
                          k_predictions,
                          negative_samples,
                          n_patches_x,
                          n_patches_y):
        """
        Creates a GIM block for each specified module in the model.
        """
        self.gim_blocks = {}
        for module, in_channels in modules_and_output_sizes.items():
            self.gim_blocks[module] = GreedyInfoMaxBlock(in_channels,
                                                         estimate_info_aggregator=self.info_estimate_aggregator,
                                                         encoding_aggregator=self.encoding_aggregator,
                                                         k_predictions=k_predictions,
                                                         negative_samples=negative_samples,
                                                         n_patches_x=n_patches_x,
                                                         n_patches_y=n_patches_y)

