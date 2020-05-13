#  Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
#  The information and source code contained herein is the
#  exclusive property of Numenta Inc.  No part of this software
#  may be used, reproduced, stored or distributed in any form,
#  without explicit written authorization from Numenta Inc.

from functools import partial

from torch import nn

from nupic.research.frameworks.backprop_structure.modules import FixedVDropConv2d
from nupic.research.frameworks.backprop_structure.networks.resnet import (
    Bottleneck,
    ResNet,
)

DEFAULT_ALPHA = (1 / 8) ** 2


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,
            alpha=DEFAULT_ALPHA):
    """3x3 convolution with padding"""
    assert groups == 1
    return FixedVDropConv2d(
        in_planes, out_planes, alpha=alpha, kernel_size=3, stride=stride,
        padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, alpha=DEFAULT_ALPHA):
    """1x1 convolution"""
    return FixedVDropConv2d(
        in_planes, out_planes, alpha=alpha, kernel_size=1, stride=stride,
        bias=False)


class FixedVDropResNet(ResNet):
    def __init__(self, alpha=DEFAULT_ALPHA, **kwargs):
        super().__init__(conv1x1_layer=partial(conv1x1, alpha=alpha),
                         conv3x3_layer=partial(conv3x3, alpha=alpha),
                         **kwargs)


def mnist_stem(out_planes):
    return nn.Conv2d(1, out_planes, kernel_size=3, stride=2,
                     padding=1, bias=False)


def imagenet_stem(out_planes):
    return nn.Conv2d(3, out_planes, kernel_size=7, stride=2,
                     padding=3, bias=False)


resnet50_fixedvdrop_imagenet = partial(
    FixedVDropResNet,
    block=Bottleneck,
    layers=[3, 4, 6, 3],
    stem_layer=imagenet_stem)


__all__ = [
    "resnet50_fixedvdrop_imagenet",
]
