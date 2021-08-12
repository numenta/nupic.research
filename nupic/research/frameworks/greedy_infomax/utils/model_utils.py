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
from nupic.research.frameworks.greedy_infomax.models.ResNetEncoder import \
    PreActBlockNoBN, PreActBottleneckNoBN, SparsePreActBlockNoBN, \
    SparsePreActBottleneckNoBN, ResNetEncoder, SparseResNetEncoder
from nupic.research.frameworks.greedy_infomax.models.BilinearInfo import \
    BilinearInfo, SparseBilinearInfo


def full_model_blockwise_config(
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        grayscale=True,
        patch_size=16,
        overlap=2,
        num_channels=None,
        block_dims=None,
):
    if block_dims is None:
        block_dims = [3, 4, 6]
    if num_channels is None:
        num_channels = [64, 128, 256]

    if resnet_50:
        block = PreActBottleneckNoBN
    else:
        block = PreActBlockNoBN

    if grayscale:
        input_dims = 1
    else:
        input_dims = 3
    encoder = []

    encoder.append(
        nn.Conv2d(input_dims, num_channels[0], kernel_size=5, stride=1, padding=2)
    )

    for idx in range(len(block_dims)):
        encoder.append(
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

small_resnet = [
    dict(
        model_class=nn.Conv2d,
        model_args=dict(in_channels=input_dims,
                        out_channels=num_channels,
                        kernel_size=5,
                        stride=1,
                        padding=2),
        init_batch_norm=None,
        checkpoint_file=None,
        load_checkpoint_args=None,
        train=True,
        save_checkpoint_file=None,
    ),
    dict(

    )

]