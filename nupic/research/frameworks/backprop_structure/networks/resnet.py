# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
# ----------------------------------------------------------------------

from functools import partial

from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet


class SingleInputChannelResNet(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(3, 3),
                               bias=False)


mnist_resnet = partial(SingleInputChannelResNet,
                       block=BasicBlock,
                       layers=[2, 2, 2, 2],
                       num_classes=10)
