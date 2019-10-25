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
from torch.autograd import Variable
from torchvision.models.utils import load_state_dict_from_url

from nupic.torch.modules import Flatten, KWinners2d

def conv1x1(in_planes, out_planes, stride=1, padding=0, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, 
        kernel_size=1, stride=stride, padding=padding, bias=bias
    )

def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, 
        kernel_size=3, stride=stride, padding=padding, bias=bias
    )

def conv5x5(in_planes, out_planes, stride=1, padding=2, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, 
        kernel_size=7, stride=stride, padding=padding, bias=bias
    )

def conv7x7(in_planes, out_planes, stride=1, padding=3, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, 
        kernel_size=7, stride=stride, padding=padding, bias=bias
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, activation_func=nn.ReLU):
        super(BasicBlock, self).__init__()

        self.regular_path = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            activation_func(planes),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes),
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
            # 1st layer
            conv1x1(in_planes, planes),
            nn.BatchNorm2d(planes),
            activation_func(planes),
            # 2nd layer
            conv3x3(planes,planes, stride=stride),
            nn.BatchNorm2d(planes),
            activation_func(planes),
            # 3rd layer
            conv1x1(planes,self.expansion * planes),
            nn.BatchNorm2d(self.expansion * planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.post_activation = activation_func(self.expansion * planes)

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
            depth=50,
            num_classes=10,
            percent_on_k_winner=1.0,
            boost_strength=1.4,
            boost_strength_factor=0.7,
            k_inference_factor=1.0,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)

        # adds kwinners
        for attr in [
            "percent_on_k_winner",
            "boost_strength",
            "boost_strength_factor",
            "k_inference_factor",
        ]:
            if type(self.__dict__[attr]) == list:
                raise ValueError(
                    """ResNet currently supports only single
                    percentage of activations for KWinners layers"""
                )

        if self.percent_on_k_winner < 0.5:
            self.activation_func = lambda out: self._kwinners(out)
        else:
            self.activation_func = lambda _: nn.ReLU()

        self.in_planes = 64
        # TODO: analyze what are these attributes used for in torchvision:
        # self.groups, self.base_width

        block, num_blocks = self._config_layers()

        self.features = nn.Sequential(
            conv7x7(3, 64, stride=2),
            nn.BatchNorm2d(64),
            self.activation_func(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AdaptiveAvgPool2d(1),
            Flatten(), # TODO: see if I still need it
        )
        self.classifier = nn.Linear(512 * block.expansion, self.num_classes)

    def _config_layers(self):
        depth_lst = [18, 34, 50, 101, 152]
        assert (
            self.depth in depth_lst
        ), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
        cf_dict = {
            "18": (BasicBlock, [2, 2, 2, 2]),
            "34": (BasicBlock, [3, 4, 6, 3]),
            "50": (Bottleneck, [3, 4, 6, 3]),
            "101": (Bottleneck, [3, 4, 23, 3]),
            "152": (Bottleneck, [3, 8, 36, 3]),
        }

        return cf_dict[str(self.depth)]

    def _kwinners(self, out):
        return KWinners2d(
            out,
            percent_on=self.percent_on_k_winner,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
            k_inference_factor=self.k_inference_factor,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation_func))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


# convenience classes

def resnet50_pretrained(config=None):
    config = config or {}
    config["depth"] = 50
    new_num_classes = config['num_classes']
    config['num_classes'] = 1000
    net = ResNet(config)

    model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    state_dict = load_state_dict_from_url(model_url, progress=True)

    def is_incompatible(layer):
        return (layer.endswith('num_batches_tracked') or 
                layer.endswith('boost_strength') or
                layer.endswith('duty_cycle'))

    # get keys, remove all num_batches_tracked
    original_state_dict = list(net.modules())[0].state_dict() 
    original_keys = original_state_dict.keys()
    original_keys = [k for k in original_keys if not is_incompatible(k)]

    # load state dict from torchvision
    assert(len(original_keys) == len(state_dict),
        "Incompatible number of layers between the created network and preloaded network")
    new_state_dict = {k:v for k,v in zip(original_keys, state_dict.values())}
    net.load_state_dict(new_state_dict, strict=False)

    # # freeze all layers
    # for param in net.parameters():
    #     param.requires_grad=False

    # replace the last layer
    classifier_shape = (net.classifier.weight.shape[1], new_num_classes)
    net.classifier = nn.Linear(*classifier_shape)

    return net

def resnet18(config=None):
    config = config or {}
    config["depth"] = 18
    return ResNet(config)


def resnet34(config=None):
    config = config or {}
    config["depth"] = 34
    return ResNet(config)

def resnet50(config=None):
    config = config or {}
    config["depth"] = 50
    return ResNet(config)

def resnet101(config=None):
    config = config or {}
    config["depth"] = 101
    return ResNet(config)


def resnet152(config=None):
    config = config or {}
    config["depth"] = 152
    return ResNet(config)


# TODO: move to tests
# if __name__ == "__main__":
#     net = ResNet(config=dict(depth=50, num_classes=10))
#     y = net(Variable(torch.randn(1, 3, 32, 32)))
#     print(y.size())
#     net = resnet101()
#     y = net(Variable(torch.randn(1, 3, 32, 32)))
#     print(y.size())
#     net = resnet152(config=dict(percent_on_k_winner=0.3))
#     y = net(Variable(torch.randn(1, 3, 32, 32)))
#     print(y.size())
