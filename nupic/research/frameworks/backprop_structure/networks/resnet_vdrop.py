#  Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc.  No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.

from functools import partial

from torch import nn

from nupic.research.frameworks.backprop_structure.modules import VDropConv2d
from nupic.research.frameworks.backprop_structure.networks.resnet import (
    Bottleneck,
    ResNet,
)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return VDropConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return VDropConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class VDropResNet(ResNet):
    def __init__(self, **kwargs):
        super().__init__(conv1x1_layer=conv1x1, conv3x3_layer=conv3x3, **kwargs)


def mnist_stem(out_planes):
    return nn.Conv2d(1, out_planes, kernel_size=3, stride=2,
                     padding=1, bias=False)


def imagenet_stem(out_planes):
    return nn.Conv2d(3, out_planes, kernel_size=7, stride=2,
                     padding=3, bias=False)


resnet50_vdrop_imagenet = partial(
    VDropResNet,
    block=Bottleneck,
    layers=[3, 4, 6, 3],
    stem_layer=imagenet_stem)


__all__ = [
    "resnet50_vdrop_imagenet",
]
