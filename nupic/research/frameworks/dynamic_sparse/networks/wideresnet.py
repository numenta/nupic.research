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

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        # resnet shortcut
        self.shortcut = nn.Sequential()
        # never for the first layer
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """
    Config hyperparamaters:
        - Batch norm is not optional
        - Dropout is one single rate, applied at all layers
        - Depth and widen_factor are specific to wide resnet architecture
    """

    def __init__(self, config):
        super(WideResNet, self).__init__()

        # update config
        defaults = dict(depth=28, widen_factor=2, num_classes=10, dropout_rate=0.3)
        defaults.update(config or {})
        self.__dict__.update(defaults)

        self.in_planes = 16

        assert (self.depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = int((self.depth - 4) / 6)  # 28-4/6 = 4
        k = self.widen_factor

        print("| Wide-Resnet %dx%d" % (self.depth, k))
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, n_stages[0])  # 1
        self.layer1 = self._wide_layer(
            WideBasic, n_stages[1], n, self.dropout_rate, stride=1
        )  # 4x2 (2 convs each)
        self.layer2 = self._wide_layer(
            WideBasic, n_stages[2], n, self.dropout_rate, stride=2
        )  # 4x2
        self.layer3 = self._wide_layer(
            WideBasic, n_stages[3], n, self.dropout_rate, stride=2
        )  # 4x2
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], self.num_classes)  # 1

        # where are the other 2?

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        # first one can define stride - downsampling
        # others are always 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # will be the size of N. In WideResnet28, will be 4
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


# net=WideResNet(28, 10, 0.3, 10)
# y = net(Variable(torch.randn(1,3,32,32)))
