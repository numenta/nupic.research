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

# adapted from https://github.com/meliketoy/wide-resnet.pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys

from nupic.torch.modules import Flatten, KWinners, KWinners2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, activation_func=nn.ReLU):
        super(BasicBlock, self).__init__()

        self.regular_path = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            activation_func(planes),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes)
            )

        self.post_activation = activation_func(planes)

    def forward(self, x):
        out = self.regular_path(x)
        out += self.shortcut(x)
        out = self.post_activation(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation_func=nn.ReLU):
        super(Bottleneck, self).__init__()
        self.regular_path = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=True),
            nn.BatchNorm2d(planes),
            activation_func(planes),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(planes),
            activation_func(planes),
            nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.expansion*planes)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.post_activation = activation_func(self.expansion*planes)

    def forward(self, x):
        out = self.regular_path(x)
        out += self.shortcut(x)
        out = self.post_activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, config=None):
        super(ResNet, self).__init__()

        # update config
        defaults = dict(
            depth=28, 
            num_classes=10,
            percent_on_k_winner=1.0,
            boost_strength=1.4,
            boost_strength_factor=0.7,
            k_inference_factor=1.0,            
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)

        # adds kwinners
        for attr in ['percent_on_k_winner', 'boost_strength', 
        'boost_strength_factor', 'k_inference_factor']:
            if type(self.__dict__[attr]) == list:
                raise ValueError("""ResNet currently supports only single 
                    percentage of activations for KWinners layers""") 

        if self.percent_on_k_winner < 0.5:
            self.activation_func = lambda out: self._kwinners(out) 
        else:
            self.activation_func = lambda _: nn.ReLU()

        self.in_planes = 16

        block, num_blocks = cfg(self.depth)

        self.features = nn.Sequential(
            conv3x3(3,16),
            nn.BatchNorm2d(16),
            self.activation_func(16),
            self._make_layer(block, 16, num_blocks[0], stride=1),
            self._make_layer(block, 32, num_blocks[1], stride=2),
            self._make_layer(block, 64, num_blocks[2], stride=2),
            nn.AdaptiveAvgPool2d((1,1))        
        )
        self.classifier = nn.Linear(64*block.expansion, self.num_classes)

    def _kwinners(self, out):
        return KWinners2d(
            out,
            percent_on=self.percent_on_k_winner,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
            k_inference_factor=self.k_inference_factor,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation_func))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# convenience classes

def resnet18(config=None):
    config = config or {}
    config['depth'] = 18
    return ResNet(config)

def resnet34(config=None):
    config = config or {}
    config['depth'] = 34
    return ResNet(config)

def resnet50(config=None):
    config = config or {}
    config['depth'] = 50
    return ResNet(config)

def resnet101(config=None):
    config = config or {}
    config['depth'] = 101
    return ResNet(config)

def resnet152(config=None):
    config = config or {}
    config['depth'] = 152
    return ResNet(config)

if __name__ == '__main__':
    net=ResNet(config=dict(depth=50, num_classes=10))
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
    net=resnet101()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
    net=resnet152(config=dict(percent_on_k_winner=0.3))
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
