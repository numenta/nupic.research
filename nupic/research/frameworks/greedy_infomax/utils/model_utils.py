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
from nupic.torch.modules import SparseWeights2d

def full_sparse_model_blockwise_config(
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
    if block_dims is None:
        block_dims = [3, 4, 6]
    if num_channels is None:
        num_channels = [64, 128, 256]

    if resnet_50:
        block = SparsePreActBottleneckNoBN
    else:
        block = SparsePreActBlockNoBN

    if grayscale:
        input_dims = 1
    else:
        input_dims = 3

    if sparsity is None:
        # reverts to dense weights
        sparsity = {"conv1": 0.01,
                    "encoder1": None,
                    "encoder2": None,
                    "encoder3": None}
    if percent_on is None:
        # reverts to relu
        percent_on = {"encoder1": None,
                      "encoder2": None,
                      "encoder3": None, }

    modules = []
    conv1_class = nn.Conv2d
    conv1_args = dict(
        in_channels=input_dims,
        out_channels=num_channels[0],
        kernel_size=5,
        stride=1,
        padding=2
    )
    if sparsity["conv1"] > 0.3:
        inner_conv_class = conv1_class
        inner_conv_args = conv1_args
        conv1_class = sparse_weights_class
        conv1_args=dict(model_blocks=[],
                        sparsity=sparsity["conv1"],
                        allow_extremes=True)

    modules.append(
        dict(
            model_class=conv1_class,
            model_args=conv1_args,
            init_batch_norm=None,
            checkpoint_file=None,
            load_checkpoint_args=None,
            train=True,
            save_checkpoint_file=None,
        )
    )
    for idx in range(len(block_dims)):
        # args
        num_blocks=[block_dims[idx]],
        filters=[num_channels[idx]],
        previous_input_dim = num_channels[0] if idx == 0 else num_channels[idx - 1],
        first_stride = 1 if idx == 0 else 2
        sparse_weights_class = sparse_weights_class,
        sparsity = sparsity[f"encoder{idx + 1}"],
        percent_on = percent_on[f"encoder{idx + 1}"],
        if sparsity is None:
            sparsity = {"block1": None,
                        "block2": None,
                        "block3": None,
                        "block4": None,
                        "block5": None,
                        "block6": None,
                        "bilinear_info": None
                        }
        if percent_on is None:
            percent_on = {"block1": None,
                          "block2": None,
                          "block3": None,
                          "block4": None,
                          "block5": None,
                          "block6": None,
                          }
        for idx in range(len(num_blocks)):
            strides = [first_stride] + [1] * (num_blocks - 1)
            in_planes = previous_input_dim
            planes=filters[idx]
            for idx, stride in enumerate(strides):
                modules.append(dict(
                    model_class=block,
                    model_args=dict(
                        in_planes=in_planes,
                        planes=planes,
                        stride=stride,
                        sparse_weights_class=sparse_weights_class,
                        sparsity=sparsity[f"block{idx + 1}"],
                        percent_on=percent_on[f"block{idx + 1}"]),
                    init_batch_norm=None,
                    checkpoint_file=None,
                    load_checkpoint_args=None,
                    train=True,
                    save_checkpoint_file=None,
                ))
                in_planes = planes * block.expansion
                first_stride = 2
        modules.append(
            dict(
                model_class=SparseBilinearInfo,
                model_args=dict(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    negative_samples=16,
                    k_predictions=5,
                    sparse_weights_class=sparse_weights_class,
                    sparsity=0.1,
                ),
                init_batch_norm=None,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=True,
                save_checkpoint_file=None,
            )
        )
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