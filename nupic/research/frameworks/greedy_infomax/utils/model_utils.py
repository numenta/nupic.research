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

import torch.nn as nn

from nupic.research.frameworks.greedy_infomax.models.bilinear_info import (
    SparseBilinearInfo,
)
from nupic.research.frameworks.greedy_infomax.models.resnet_encoder import (
    SparsePreActBlockNoBN,
    SparsePreActBottleneckNoBN,
)
from nupic.research.frameworks.greedy_infomax.models.utility_layers import (
    EmitEncoding,
    GradientBlock,
    PatchifyInputs,
    SparseConv2d,
)
from nupic.torch.modules import SparseWeights2d



def _make_layer_config(
        block,
        num_blocks,
        planes,
        in_planes,
        stride=1,
        sparse_weights_class=SparseWeights2d,
        sparsity=None,
        percent_on=None,):
    layer_config = []
    strides = [stride] + [1] * (num_blocks - 1)
    if sparsity is None:
        sparsity = {}
    if percent_on is None:
        percent_on = {}
    for i, stride in enumerate(strides):
        layer_config.append(
            dict(
                model_class=block,
                model_args=dict(
                    in_planes=in_planes,
                    planes=planes,
                    stride=stride,
                    sparse_weights_class=sparse_weights_class,
                    sparsity=sparsity.get(f"block{i + 1}", None),
                    percent_on=percent_on.get(f"block{i + 1}", None),
                ),
                init_batch_norm=False,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=True,
                save_checkpoint_file=None,
            )
        )
        in_planes = planes * block.expansion
    return layer_config

def full_sparse_model_blockwise_config(
    negative_samples=16,
    k_predictions=5,
    resnet_50=False,
    block_dims=None,
    num_channels=None,
    grayscale=True,
    sparse_weights_class=SparseWeights2d,
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
        sparsity = {"conv1": 0.01, "encoder1": None, "encoder2": None, "encoder3": None}
    if percent_on is None:
        # reverts to relu
        percent_on = {"encoder1": None, "encoder2": None, "encoder3": None}

    modules = []
    modules.append(
        dict(
            model_class=PatchifyInputs,
            model_args=dict(patch_size=patch_size, overlap=overlap),
            init_batch_norm=False,
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
        padding=2,
    )
    if sparsity["conv1"] > 0.3:
        conv1_class = SparseConv2d
        conv1_args.update(
            dict(
                sparse_weights_class=sparse_weights_class,
                sparsity=sparsity["conv1"],
                allow_extremes=True,
            )
        )
    modules.append(
        dict(
            model_class=conv1_class,
            model_args=conv1_args,
            init_batch_norm=False,
            checkpoint_file=None,
            load_checkpoint_args=None,
            train=True,
            save_checkpoint_file=None,
        )
    )
    for idx in range(len(block_dims)):
        # args
        num_blocks = block_dims[idx]
        filters = num_channels[idx]
        previous_input_dim = num_channels[0] if idx == 0 else num_channels[idx - 1]
        first_stride = 1 if idx == 0 else 2
        encoder_sparsity = sparsity[f"encoder{idx + 1}"]
        encoder_percent_on = percent_on[f"encoder{idx + 1}"]
        in_planes = previous_input_dim if idx==0 else previous_input_dim * block.expansion
        modules.extend(_make_layer_config(
            block,
            num_blocks,
            filters,
            in_planes,
            sparse_weights_class=SparseWeights2d,
            sparsity=encoder_sparsity,
            percent_on=encoder_percent_on,
            stride=first_stride,
        ))
        modules.append(
            dict(
                model_class=SparseBilinearInfo,
                model_args=dict(
                    in_channels=filters * block.expansion,
                    out_channels=filters * block.expansion,
                    negative_samples=negative_samples,
                    k_predictions=k_predictions,
                    sparse_weights_class=sparse_weights_class,
                    sparsity=0.1,
                ),
                init_batch_norm=False,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=True,
                save_checkpoint_file=None,
            )
        )
        modules.append(
            dict(
                model_class=EmitEncoding,
                model_args=dict(channels=filters * block.expansion),
                init_batch_norm=False,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=False,
                save_checkpoint_file=None,
            )
        )

        modules.append(
            dict(
                model_class=GradientBlock,
                model_args=dict(),
                init_batch_norm=False,
                checkpoint_file=None,
                load_checkpoint_args=None,
                train=False,
                save_checkpoint_file=None,
            )
        )
    return modules




#predefined configs
full_resnet_50 = full_sparse_model_blockwise_config(resnet_50=True)

full_resnet = full_sparse_model_blockwise_config(resnet_50=False)
small_resnet = full_sparse_model_blockwise_config()[:8]
full_sparse_resnet_34 = full_sparse_model_blockwise_config(
    sparsity=dict(
        conv1=0.01,  # dense
        encoder1=dict(
            block1=dict(conv1=0.7, conv2=0.7),
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.1,  # dense weights
        ),
        encoder2=dict(
            block1=dict(conv1=0.7, conv2=0.7, shortcut=0.01),
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            block4=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.01,
        ),
        encoder3=dict(
            block1=dict(conv1=0.7, conv2=0.7, shortcut=0.01),  # dense
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            block4=dict(conv1=0.7, conv2=0.7),
            block5=dict(conv1=0.7, conv2=0.7),
            block6=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.01,  # dense
        ),
    ),
    num_channels=[117, 233, 467],
)
small_sparse_resnet = full_sparse_model_blockwise_config(
    sparsity=dict(
        conv1=0.01,  # dense
        encoder1=dict(
            block1=dict(conv1=0.7, conv2=0.7),
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.1,  # dense weights
        ),
        encoder2=dict(
            block1=dict(conv1=0.7, conv2=0.7, shortcut=0.01),
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            block4=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.01,
        ),
        encoder3=dict(
            block1=dict(conv1=0.7, conv2=0.7, shortcut=0.01),  # dense
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            block4=dict(conv1=0.7, conv2=0.7),
            block5=dict(conv1=0.7, conv2=0.7),
            block6=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.01,  # dense
        ),
    ),
    num_channels=[117, 233, 467],
)[:8]
