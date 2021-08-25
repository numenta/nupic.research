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
from nupic.research.frameworks.greedy_infomax.models import FullModel
from nupic.research.frameworks.greedy_infomax.models.ResNetEncoder import \
    PreActBlockNoBN, PreActBottleneckNoBN, SparsePreActBlockNoBN, \
    SparsePreActBottleneckNoBN, ResNetEncoder, SparseResNetEncoder
from nupic.research.frameworks.greedy_infomax.models.BilinearInfo import \
    BilinearInfo, SparseBilinearInfo
from nupic.research.frameworks.greedy_infomax.models.UtilityLayers import \
    GradientBlock, EmitEncoding, PatchifyInputs, SparseConv2d

def full_sparse_model_blockwise_config(
        negative_samples=16,
        k_predictions=5,
        resnet_50=False,
        block_dims=None,
        num_channels=None,
        grayscale=True,
        sparse_weights_class=SparseConv2d,
        patch_size=16,
        overlap=2,
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
    modules.append(
        dict(
            model_class=PatchifyInputs,
            model_args=dict(
                patch_size=patch_size,
                overlap=overlap,
            ),
            init_batch_norm=None,
            checkpoint_file=None,
            load_checkpoint_args=None,
            train=True,
            save_checkpoint_file=None,
        )
    )

    conv1_class = nn.Conv2d
    conv1_args = dict(
        in_channels=input_dims,
        out_channels=num_channels[0],
        kernel_size=5,
        stride=1,
        padding=2
    )
    if sparsity["conv1"] > 0.3:
        conv1_class = SparseConv2d
        conv1_args.update(
            dict(
                sparse_weights_class=sparse_weights_class,
                sparsity=sparsity["conv1"],
                allow_extremes=True
            )
        )
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
        num_blocks=block_dims[idx],
        filters=num_channels[idx],
        previous_input_dim = num_channels[0] if idx == 0 else num_channels[idx - 1],
        first_stride = 1 if idx == 0 else 2
        encoder_sparsity = sparsity[f"encoder{idx + 1}"]
        encoder_percent_on = percent_on[f"encoder{idx + 1}"]
        if encoder_sparsity is None:
            encoder_sparsity = {"block1": None,
                        "block2": None,
                        "block3": None,
                        "block4": None,
                        "block5": None,
                        "block6": None,
                        "bilinear_info": None
                        }
        if encoder_percent_on is None:
            encoder_percent_on = {"block1": None,
                          "block2": None,
                          "block3": None,
                          "block4": None,
                          "block5": None,
                          "block6": None,
                          }
        strides = [first_stride] + [1] * (num_blocks[0] - 1)
        in_planes = previous_input_dim
        planes=filters[0]
        for idx, stride in enumerate(strides):
            modules.append(dict(
                model_class=block,
                model_args=dict(
                    in_planes=in_planes,
                    planes=planes,
                    stride=stride,
                    sparse_weights_class=sparse_weights_class,
                    sparsity=encoder_sparsity[f"block{idx + 1}"],
                    percent_on=encoder_percent_on[f"block{idx + 1}"]),
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
                    in_channels=in_planes,
                    out_channels=in_planes,
                    negative_samples=negative_samples,
                    k_predictions=k_predictions,
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
        modules.append(
            dict(
                model_class=GradientBlock,
                model_args=dict(),
                init_batch_norm=None,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=True,
                save_checkpoint_file=None,
            )
        )
        modules.append(
            dict(
                model_class=EmitEncoding,
                model_args=dict(),
                init_batch_norm=None,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=True,
                save_checkpoint_file=None,
            )
        )
    return modules



full_sparse_resnet = full_sparse_model_blockwise_config()
small_sparse_resnet = full_sparse_resnet[:8]
