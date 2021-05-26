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

import torch
import torch.nn as nn

from nupic.research.frameworks.greedy_infomax.models import (
    PreActBlockNoBN,
    PreActBottleneckNoBN,
    ResNetEncoder,
    SparsePreActBlockNoBN,
    SparsePreActBottleneckNoBN,
    SparseResNetEncoder,
)
from nupic.research.frameworks.greedy_infomax.utils import model_utils


class FullVisionModel(torch.nn.Module):
    """
    A modified version of ResNet to compute patch-wise representations. This model
    is the encoder in self-supervised experiments and does not include a built in
    classifier. As an encoder, this module utilizes a .forward() for unsupervised
    training and a .encode() to produce patch-level representations. The BilinearInfo
    modules are thus only called during .forward() to prevent wasted computation.

    :param negative_samples: number of negative samples to contrast per positive sample
    :param k_predictions: number of prediction steps to compare positive examples.
                          For example, if k_predictions is 5 and skip_step is 1,
                          then this module will compare z_{t} with z_{t+2}...z{t+6}.
    :param resnet_50: If True, uses the full ResNet50 model. If False, uses the
                      smaller Resnet34.
    :param grayscale: This parameter should match the transform used on the dataset.
                      This does not actively grayscale the incoming data, but rather
                      informs the model to use either 1 or 3 channels.
    :param patch_size: The size of patches to split each image along both the x and
                       y dimensions.
    :param overlap: number of pixels of overlap between neighboring patches
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        grayscale=True,
        patch_size=16,
        overlap=2,
    ):
        super().__init__()
        self.negative_samples = negative_samples
        self.k_predictions = k_predictions
        self.patch_size = patch_size
        self.overlap = overlap
        print("Contrasting against ", self.negative_samples, " negative samples")

        block_dims = [3, 4, 6]
        num_channels = [64, 128, 256]

        self.encoder = nn.ModuleList([])

        if resnet_50:
            self.block = PreActBottleneckNoBN
        else:
            self.block = PreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        for idx in range(len(block_dims)):
            self.encoder.append(
                ResNetEncoder(
                    self.block,
                    [block_dims[idx]],
                    [num_channels[idx]],
                    idx,
                    input_dims=input_dims,
                    k_predictions=self.k_predictions,
                    negative_samples=self.negative_samples,
                )
            )

    def forward(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        # Save positive/contrastive samples for each encoder block
        log_f_module_list, true_f_module_list = [], []
        for module in self.encoder:
            # log_f_list and true_f_list each have k_predictions elements
            log_f_list, true_f_list, z = module(x, n_patches_x, n_patches_y)
            log_f_module_list.append(log_f_list)
            true_f_module_list.append(true_f_list)
            # Detach x to make sure no gradients are flowing in between modules
            x = z.detach()
        # Lists of lists: each list has num_modules internal lists, with each
        # internal list containing k_predictions elements
        return log_f_module_list, true_f_module_list

    def encode(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        # Compute encoded patch-level representation for each encoder block
        for module in self.encoder:
            # no need to detach between modules as .encode() will only be called
            # under a torch.no_grad() scope
            x, out = module.encode(x, n_patches_x, n_patches_y)
        # Return patch-level representation from the last block
        return out


class SparseFullVisionModel(FullVisionModel):
    """
    A sparse weights version of the above FullVisionModel.
    :param negative_samples: number of negative samples to contrast per positive sample
    :param k_predictions: number of prediction steps to compare positive examples.
                          For example, if k_predictions is 5 and skip_step is 1,
                          then this module will compare z_{t} with z_{t+2}...z{t+6}.
    :param resnet_50: If True, uses the full ResNet50 model. If False, uses the
                      smaller Resnet34.
    :param grayscale: This parameter should match the transform used on the dataset.
                      This does not actively grayscale the incoming data, but rather
                      informs the model to use either 1 or 3 channels.
    :param patch_size: The size of patches to split each image along both the x and
                       y dimensions.
    :param overlap: number of pixels of overlap between neighboring patches
    :param sparsity: a list of sparsity values, one for each ResNetEncoder
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        grayscale=True,
        patch_size=16,
        overlap=2,
        weight_sparsity=None,
        activation_sparsity=None,
    ):
        super(SparseFullVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
        )
        if weight_sparsity is None:
            weight_sparsity = [0.5, 0.5, 0.5]
        if activation_sparsity is None:
            activation_sparsity = [0.5, 0.5, 0.5]

        block_dims = [3, 4, 6]
        num_channels = [64, 128, 256]

        self.encoder = nn.ModuleList([])

        if resnet_50:
            self.block = SparsePreActBottleneckNoBN
        else:
            self.block = SparsePreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        for idx in range(len(block_dims)):
            self.encoder.append(
                SparseResNetEncoder(
                    self.block,
                    [block_dims[idx]],
                    [num_channels[idx]],
                    idx,
                    input_dims=input_dims,
                    k_predictions=self.k_predictions,
                    negative_samples=self.negative_samples,
                    weight_sparsity=weight_sparsity[idx],
                    activation_sparsity=activation_sparsity[idx]
                )
            )
