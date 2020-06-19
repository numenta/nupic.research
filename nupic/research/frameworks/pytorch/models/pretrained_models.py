# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
# summary
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Pretrained models imported from external repositories.

For faster loading in worker nodes, use a shared space for head node and workers
to access pretrained weights by setting the environment variable TORCH_HOME.
Preload the weights on head node when running for the first time.

Example:
export TORCH_HOME=/home/ec2-user/nta/results/torch

In Python console:
from nupic.research.frameworks.pytorch.models import <model name>
<model name>()

Note:
- models larger than Resnet50 will not fit the GPU with the regular batch size
- if SSL certificate error, run in python console prior to downloading models:
  import ssl; ssl._create_default_https_context = ssl._create_unverified_context
"""

import pretrainedmodels
import torch


def resnet50_swsl():
    """
    From: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models

    Regular Resnet50 network trained in semi-weakly supervised fashion.

    "Semi-weakly" supervised (SWSL) ImageNet models are pre-trained on 940 million
    public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed
    by fine-tuning on ImageNet1K dataset. In this case, the associated hashtags are
    only used for building a better teacher model.

    During training the student model, those hashtags are ingored and the student model
    is pretrained with a subset of 64M images selected by the teacher model from the
    same 940 million public image dataset.

    Params 25M, GFLOPs 4, Top-1 acc 81.2, Top-5 acc 96.0
    """
    return torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl"
    )

def resnext50_32x4d_swsl():
    """
    From: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models

    Residual networks with grouped convolutional layers,
    trained in semi-weakly supervised fashion.
    ResNext: https://arxiv.org/pdf/1611.05431.pdf


    "Semi-weakly" supervised (SWSL) ImageNet models are pre-trained on 940 million
    public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed
    by fine-tuning on ImageNet1K dataset. In this case, the associated hashtags are
    only used for building a better teacher model.

    During training the student model, those hashtags are ingored and the student model
    is pretrained with a subset of 64M images selected by the teacher model from the
    same 940 million public image dataset.

    Params 25M, GFLOPs 4, Top-1 acc 82.2, Top-5 acc 96.3
    """
    return torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnext50_32x4d_swsl"
    )

def resnext101_32x16d_wsl():
    """
    From: https://github.com/facebookresearch/WSL-Images

    Residual networks with grouped convolutional layers
    ResNeXt-101 32×Cd, which has 101 layers, 32 groups, and group widths C of:
    4 (8B multiply-add FLOPs, 43M parameters), 8 (16B,88M),
    16 (36B, 193M), 32 (87B, 466M), and 48 (153B, 829M).

    Pre-trained in weakly-supervised fashion on 940 million public images with
    1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning
    on ImageNet1K dataset

    Params 193M, GFLOPs 36, Top-1 acc 84.2, Top-5 acc 97.2
    """
    return torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x16d_wsl")


def resnext101_32x48d_wsl():
    """
    From: https://github.com/facebookresearch/WSL-Images

    Residual networks with grouped convolutional layers
    ResNeXt-101 32×Cd, which has 101 layers, 32 groups, and group widths C of:
    4 (8B multiply-add FLOPs, 43M parameters), 8 (16B,88M),
    16 (36B, 193M), 32 (87B, 466M), and 48 (153B, 829M).

    Pre-trained in weakly-supervised fashion on 940 million public images with
    1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning
    on ImageNet1K dataset

    Params 829M, GFLOPs 153, Top-1 acc 85.4, Top-5 acc 97.6
    """
    return torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x48d_wsl")


def se_resnext50_32x4d():
    """
    From: https://github.com/Cadene/pretrained-models.pytorch
    From: https://github.com/hujie-frank/SENet

    Residual networks with grouped convolutional layers and squeeze & excitation blocks
    ResNext: https://arxiv.org/pdf/1611.05431.pdf
    Squeeze and Excitation: https://arxiv.org/abs/1709.01507

    Params 27M, GFLOPs 4.25, Top-1 acc 79.076, Top-5 acc 94.434
    """
    return pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")


def se_resnet50():
    """
    From: https://github.com/Cadene/pretrained-models.pytorch
    From: https://github.com/hujie-frank/SENet

    Residual networks with squeeze & excitation blocks
    Squeeze and Excitation: https://arxiv.org/abs/1709.01507

    Params ~25M, size 107MB, Top-1 acc 77.636, Top-5 acc 93.752
    """
    return pretrainedmodels.se_resnet50(num_classes=1000, pretrained="imagenet")


def xception():
    """
    From: https://github.com/Cadene/pretrained-models.pytorch

    Interpretation of Inception modules in convolutional neural networks as being
    an intermediate step in-between regular convolution and the depthwise separable
    convolution operation
    https://arxiv.org/abs/1610.02357

    Params 24M, size 87MB, Top-1 acc 79.000, Top-5 acc 94.500
    """
    return pretrainedmodels.xception(num_classes=1000, pretrained="imagenet")
