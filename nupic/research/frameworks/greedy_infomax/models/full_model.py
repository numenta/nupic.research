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

from nupic.research.frameworks.backprop_structure.modules import (
    MaskedVDropCentralData,
    VDropConv2d,
)
from nupic.research.frameworks.greedy_infomax.models.resnet_encoder import (
    PreActBlockNoBN,
    PreActBottleneckNoBN,
    ResNetEncoder,
    SparsePreActBlockNoBN,
    SparsePreActBottleneckNoBN,
    SparseResNetEncoder,
    SuperGreedySparseResNetEncoder,
    VDropSparsePreActBlockNoBN,
    VDropSparsePreActBottleneckNoBN,
    VDropSparseResNetEncoder,
)
from nupic.research.frameworks.greedy_infomax.utils import model_utils
from nupic.torch.modules import SparseWeights2d


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
        num_channels=None,
        block_dims=None,
    ):
        super().__init__()
        self.negative_samples = negative_samples
        self.k_predictions = k_predictions
        self.patch_size = patch_size
        self.overlap = overlap
        print("Contrasting against ", self.negative_samples, " negative samples")

        if block_dims is None:
            block_dims = [3, 4, 6]
        if num_channels is None:
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

        self.encoder.append(
            nn.Conv2d(input_dims, num_channels[0], kernel_size=5, stride=1, padding=2)
        )

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
                    previous_input_dim=num_channels[0]
                    if idx == 0
                    else num_channels[idx - 1],
                    first_stride=1 if idx == 0 else 2,
                )
            )

    def forward(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        x = self.encoder[0](x)
        # Save positive/contrastive samples for each encoder block
        log_f_module_list, true_f_module_list = [], []
        for module in self.encoder[1:]:
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
        x = self.encoder[0](x)
        # Compute encoded patch-level representation for each encoder block
        for module in self.encoder[1:]:
            # no need to detach between modules as .encode() will only be called
            # under a torch.no_grad() scope
            x, out = module.encode(x, n_patches_x, n_patches_y)
        # Return patch-level representation from the last block
        return out


class SparseFullVisionModel(FullVisionModel):
    """
    A version of the above FullVisionModel that uses sparse weights and activations.
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
    :param percent_on:  a list of 3 values between (0, 1) which represent the
    percentage of units on in each block of the ResNetEncoder
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        sparse_weights_class=SparseWeights2d,
        sparsity=None,
        percent_on=None,
    ):
        super(SparseFullVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
        )
        if sparsity is None:
            # reverts to dense weights
            sparsity = {
                "conv1": 0.01,
                "encoder1": None,
                "encoder2": None,
                "encoder3": None,
            }
        if percent_on is None:
            # reverts to relu
            percent_on = {"encoder1": None, "encoder2": None, "encoder3": None}
        if block_dims is None:
            block_dims = [3, 4, 6]
        if num_channels is None:
            num_channels = [64, 128, 256]

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder = nn.ModuleList([])
        if sparsity["conv1"] > 0.3:
            self.encoder.append(
                sparse_weights_class(
                    nn.Conv2d(
                        input_dims, num_channels[0], kernel_size=5, stride=1, padding=2
                    ),
                    sparsity=sparsity["conv1"],
                    allow_extremes=True,
                )
            )
        else:
            self.encoder.append(
                nn.Conv2d(
                    input_dims, num_channels[0], kernel_size=5, stride=1, padding=2
                )
            )

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
                    sparse_weights_class=sparse_weights_class,
                    sparsity=sparsity[f"encoder{idx+1}"],
                    percent_on=percent_on[f"encoder{idx+1}"],
                    previous_input_dim=num_channels[0]
                    if idx == 0
                    else num_channels[idx - 1],
                    first_stride=1 if idx == 0 else 2,
                )
            )


class VDropSparseFullVisionModel(FullVisionModel):
    """
    A version of the above FullVisionModel that uses global variational dropout to
    achieve sparse weights and k-winners modules for sparse activations. Note that
    the weight sparsity is controlled by the PruneLowSNR mixin config.

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
    :param percent_on:  a list of 3 values between (0, 1) which represent the
    percentage of units on in each block of the ResNetEncoder
    :param central_data: a VDropCentralData module for intializing VDrop
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        percent_on=None,
    ):
        super(VDropSparseFullVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
        )
        if percent_on is None:
            # reverts to relu
            percent_on = {"encoder1": None, "encoder2": None, "encoder3": None}
        if block_dims is None:
            block_dims = [3, 4, 6]
        if num_channels is None:
            num_channels = [64, 128, 256]

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.vdrop_central_data = MaskedVDropCentralData()

        self.encoder = nn.ModuleList([])
        self.encoder.append(
            VDropConv2d(
                input_dims,
                num_channels[0],
                kernel_size=5,
                central_data=self.vdrop_central_data,
                stride=1,
                padding=2,
            )
        )

        if resnet_50:
            self.block = VDropSparsePreActBottleneckNoBN
        else:
            self.block = VDropSparsePreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        for idx in range(len(block_dims)):
            self.encoder.append(
                VDropSparseResNetEncoder(
                    self.block,
                    [block_dims[idx]],
                    [num_channels[idx]],
                    idx,
                    input_dims=input_dims,
                    k_predictions=self.k_predictions,
                    negative_samples=self.negative_samples,
                    percent_on=percent_on[f"encoder{idx+1}"],
                    previous_input_dim=num_channels[0]
                    if idx == 0
                    else num_channels[idx - 1],
                    first_stride=1 if idx == 0 else 2,
                    central_data=self.vdrop_central_data,
                )
            )
        self.vdrop_central_data.finalize()

    def forward(self, *args, **kwargs):
        self.vdrop_central_data.compute_forward_data()
        ret = super().forward(*args, **kwargs)
        self.vdrop_central_data.clear_forward_data()
        return ret

    def encode(self, *args, **kwargs):
        self.vdrop_central_data.compute_forward_data()
        ret = super().encode(*args, **kwargs)
        self.vdrop_central_data.clear_forward_data()
        return ret

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.vdrop_central_data = self.vdrop_central_data.to(*args, **kwargs)
        return ret


class SmallVisionModel(torch.nn.Module):
    """
    A smaller version of the above FullVisionModel which only uses the first
    ResNetEncoder block.

    :param negative_samples: number of negative samples to contrast per positive sample
    :param k_predictions: number of prediction steps to compare positive examples.
                          For example, if k_predictions is 5 and skip_step is 1,
                          then this module will compare z_{t} with z_{t+2}...z{t+6}.
    :param resnet_50: If True, uses the full ResNet50 model. If False, uses the
                      smaller ResNet34. Defaults to ResNet34.
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
        num_channels=64,
        block_dims=3,
    ):
        super(SmallVisionModel, self).__init__()
        self.negative_samples = negative_samples
        self.k_predictions = k_predictions
        self.patch_size = patch_size
        self.overlap = overlap
        print("Contrasting against ", self.negative_samples, " negative samples")

        self.encoder = nn.ModuleList([])

        if resnet_50:
            self.block = PreActBottleneckNoBN
        else:
            self.block = PreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder.append(
            nn.Conv2d(input_dims, num_channels, kernel_size=5, stride=1, padding=2)
        )
        self.encoder.append(
            ResNetEncoder(
                self.block,
                [block_dims],
                [num_channels],
                0,
                input_dims=input_dims,
                k_predictions=self.k_predictions,
                negative_samples=self.negative_samples,
                previous_input_dim=num_channels,
                first_stride=1,
            )
        )

    def forward(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        x = self.encoder[0](x)
        # Save positive/contrastive samples for each encoder block
        log_f_module_list, true_f_module_list = [], []
        for module in self.encoder[1:]:
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
        x = self.encoder[0](x)
        # Compute encoded patch-level representation for each encoder block
        for module in self.encoder[1:]:
            # no need to detach between modules as .encode() will only be called
            # under a torch.no_grad() scope
            x, out = module.encode(x, n_patches_x, n_patches_y)
        # Return patch-level representation from the last block
        return out


class SparseSmallVisionModel(SmallVisionModel):
    """
    A version of the above SmallVisionModel that uses sparse weights and activations.
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
    :param percent_on:  a list of 3 values between (0, 1) which represent the
    percentage of units on in each block of the ResNetEncoder
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        sparse_weights_class=SparseWeights2d,
        sparsity=None,
        percent_on=None,
    ):
        super(SparseSmallVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
        )
        if sparsity is None:
            # reverts to dense weights
            sparsity = {"conv1": 0.01, "encoder1": None}
        if percent_on is None:
            # reverts to relu
            percent_on = {"encoder1": None}
        if block_dims is None:
            block_dims = 3
        if num_channels is None:
            num_channels = 64

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder = nn.ModuleList([])
        if sparsity["conv1"] > 0.3:
            self.encoder.append(
                sparse_weights_class(
                    nn.Conv2d(
                        input_dims, num_channels, kernel_size=5, stride=1, padding=2
                    ),
                    sparsity=sparsity["conv1"],
                    allow_extremes=True,
                )
            )
        else:
            self.encoder.append(
                nn.Conv2d(input_dims, num_channels, kernel_size=5, stride=1, padding=2)
            )

        if resnet_50:
            self.block = SparsePreActBottleneckNoBN
        else:
            self.block = SparsePreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder.append(
            SparseResNetEncoder(
                self.block,
                [block_dims],
                [num_channels],
                0,
                input_dims=input_dims,
                k_predictions=self.k_predictions,
                negative_samples=self.negative_samples,
                sparse_weights_class=sparse_weights_class,
                sparsity=sparsity["encoder1"],
                percent_on=percent_on["encoder1"],
                previous_input_dim=num_channels,
                first_stride=1,
            )
        )


class WrappedSparseSmallVisionModel(SparseSmallVisionModel):
    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        sparse_weights_class=SparseWeights2d,
        sparsity=None,
        percent_on=None,
    ):
        super(WrappedSparseSmallVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            block_dims=block_dims,
            num_channels=num_channels,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
            sparse_weights_class=sparse_weights_class,
            sparsity=dict(
                conv1=0.01,  # dense
                encoder1=dict(
                    block1=dict(conv1=sparsity, conv2=sparsity),
                    block2=dict(conv1=sparsity, conv2=sparsity),
                    block3=dict(conv1=sparsity, conv2=sparsity),
                    bilinear_info=0.01,  # dense weights
                ),
            ),
            percent_on=dict(
                encoder1=dict(
                    block1=dict(nonlinearity1=percent_on, nonlinearity2=percent_on),
                    block2=dict(nonlinearity1=percent_on, nonlinearity2=percent_on),
                    block3=dict(nonlinearity1=percent_on, nonlinearity2=percent_on),
                )
            ),
        )


class SuperGreedySparseSmallVisionModel(SparseSmallVisionModel):
    """
    A version of the above SmallVisionModel that uses sparse weights and activations.
    Also, this uses the GreedyInfoMax loss on a layer-by-layer basis.
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
    :param percent_on:  a list of 3 values between (0, 1) which represent the
    percentage of units on in each block of the ResNetEncoder
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        sparse_weights_class=SparseWeights2d,
        sparsity=None,
        percent_on=None,
    ):
        super(SuperGreedySparseSmallVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
        )
        if sparsity is None:
            # reverts to dense weights
            sparsity = {"conv1": 0.01, "encoder1": None}
        if percent_on is None:
            # reverts to relu
            percent_on = {"encoder1": None}
        if block_dims is None:
            block_dims = 3
        if num_channels is None:
            num_channels = 64

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder = nn.ModuleList([])
        if sparsity["conv1"] > 0.3:
            self.encoder.append(
                sparse_weights_class(
                    nn.Conv2d(
                        input_dims, num_channels, kernel_size=5, stride=1, padding=2
                    ),
                    sparsity=sparsity["conv1"],
                    allow_extremes=True,
                )
            )
        else:
            self.encoder.append(
                nn.Conv2d(input_dims, num_channels, kernel_size=5, stride=1, padding=2)
            )

        if resnet_50:
            self.block = SparsePreActBottleneckNoBN
        else:
            self.block = SparsePreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder.append(
            SuperGreedySparseResNetEncoder(
                self.block,
                [block_dims],
                [num_channels],
                0,
                input_dims=input_dims,
                k_predictions=self.k_predictions,
                negative_samples=self.negative_samples,
                sparse_weights_class=sparse_weights_class,
                sparsity=sparsity["encoder1"],
                percent_on=percent_on["encoder1"],
                previous_input_dim=num_channels,
                first_stride=1,
            )
        )

    def forward(self, x):
        # Patchify inputs
        x, n_patches_x, n_patches_y = model_utils.patchify_inputs(
            x, self.patch_size, self.overlap
        )
        x = self.encoder[0](x)
        # Save positive/contrastive samples for each encoder block
        log_f_module_list, true_f_module_list, z = self.encoder[1](
            x, n_patches_x, n_patches_y
        )
        # Lists of lists: each list has num_modules internal lists, with each
        # internal list containing k_predictions elements
        return log_f_module_list, true_f_module_list


class WrappedSuperGreedySmallSparseVisionModel(SuperGreedySparseSmallVisionModel):
    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        sparse_weights_class=SparseWeights2d,
        sparsity=None,
        percent_on=None,
    ):
        super(WrappedSuperGreedySmallSparseVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            block_dims=block_dims,
            num_channels=num_channels,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
            sparse_weights_class=sparse_weights_class,
            sparsity=dict(
                conv1=0.01,  # dense
                encoder1=dict(
                    block1=dict(conv1=sparsity, conv2=sparsity),
                    block2=dict(conv1=sparsity, conv2=sparsity),
                    block3=dict(conv1=sparsity, conv2=sparsity),
                    bilinear_info=0.01,  # dense weights
                ),
            ),
            percent_on=dict(
                encoder1=dict(
                    block1=dict(nonlinearity1=percent_on, nonlinearity2=percent_on),
                    block2=dict(nonlinearity1=percent_on, nonlinearity2=percent_on),
                    block3=dict(nonlinearity1=percent_on, nonlinearity2=percent_on),
                )
            ),
        )


class VDropSparseSmallVisionModel(SmallVisionModel):
    """
    A version of the above SmallVisionModel that uses global variational dropout to
    achieve sparse weights and k-winners modules for sparse activations. Note that
    the weight sparsity is controlled by the PruneLowSNR mixin config.

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
    :param percent_on:  a list of 3 values between (0, 1) which represent the
    percentage of units on in each block of the ResNetEncoder
    :param central_data: a VDropCentralData module for intializing VDrop
    """

    def __init__(
        self,
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        patch_size=16,
        overlap=2,
        percent_on=None,
    ):
        super(VDropSparseSmallVisionModel, self).__init__(
            negative_samples=negative_samples,
            k_predictions=k_predictions,
            resnet_50=resnet_50,
            grayscale=grayscale,
            patch_size=patch_size,
            overlap=overlap,
        )
        if percent_on is None:
            # reverts to relu
            percent_on = {"encoder1": None}
        if block_dims is None:
            block_dims = 3
        if num_channels is None:
            num_channels = 64

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.vdrop_central_data = MaskedVDropCentralData()

        self.encoder = nn.ModuleList([])
        self.encoder.append(
            VDropConv2d(
                input_dims,
                num_channels,
                kernel_size=5,
                central_data=self.vdrop_central_data,
                stride=1,
                padding=2,
            )
        )

        if resnet_50:
            self.block = VDropSparsePreActBottleneckNoBN
        else:
            self.block = VDropSparsePreActBlockNoBN

        if grayscale:
            input_dims = 1
        else:
            input_dims = 3

        self.encoder.append(
            VDropSparseResNetEncoder(
                self.block,
                [block_dims],
                [num_channels],
                0,
                input_dims=input_dims,
                k_predictions=self.k_predictions,
                negative_samples=self.negative_samples,
                percent_on=percent_on["encoder1"],
                previous_input_dim=num_channels,
                first_stride=1,
                central_data=self.vdrop_central_data,
            )
        )
        self.vdrop_central_data.finalize()

    def forward(self, *args, **kwargs):
        self.vdrop_central_data.compute_forward_data()
        ret = super().forward(*args, **kwargs)
        self.vdrop_central_data.clear_forward_data()
        return ret

    def encode(self, *args, **kwargs):
        self.vdrop_central_data.compute_forward_data()
        ret = super().encode(*args, **kwargs)
        self.vdrop_central_data.clear_forward_data()
        return ret

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.vdrop_central_data = self.vdrop_central_data.to(*args, **kwargs)
        return ret
