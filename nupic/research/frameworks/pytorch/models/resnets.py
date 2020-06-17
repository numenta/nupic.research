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
An extensible ResNet class

Originally based on torchvision Resnet @
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py and
https://github.com/meliketoy/wide-resnet.pytorch/ but with many modifications.
"""

from collections import OrderedDict
from functools import partial

import torch.nn as nn

from nupic.torch.modules import Flatten


class BasicBlock(nn.Module):
    """Default block for ResNets with < 50 layers."""

    expansion = 1
    conv_keys = ["conv3x3_1", "conv3x3_2", "shortcut"]
    act_keys = ["act1", "act2"]
    norm_keys = ["bn1", "bn2", "shortcut"]

    def __init__(self, in_planes, planes, stride, conv_layer, conv_args,
                 act_layer, act_args, norm_layer, norm_args):
        super(BasicBlock, self).__init__()

        self.regular_path = nn.Sequential(OrderedDict([
            ("conv1", conv_layer(in_planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False,
                                 **conv_args["conv3x3_1"])),
            ("bn1", norm_layer(planes, **norm_args["bn1"])),
            ("act1", act_layer(planes, **act_args["act1"])),
            ("conv2", conv_layer(planes, planes, kernel_size=3, padding=1,
                                 bias=False, **conv_args["conv3x3_2"])),
            ("bn2", norm_layer(planes, **norm_args["bn2"])),
        ]))

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", conv_layer(in_planes, planes, kernel_size=1,
                                    stride=stride, bias=False,
                                    **conv_args["shortcut"])),
                ("bn", norm_layer(planes, **norm_args["shortcut"])),
            ]))
        else:
            self.shortcut = nn.Identity()

        self.post_activation = act_layer(planes, **act_args["act2"])

    def forward(self, x):
        out = self.regular_path(x)
        out += self.shortcut(x)
        out = self.post_activation(out)
        return out


class Bottleneck(nn.Module):
    """Default block for ResNets with >= 50 layers."""

    expansion = 4
    conv_keys = ["conv1x1_1", "conv3x3_2", "conv1x1_3", "shortcut"]
    act_keys = ["act1", "act2", "act3"]
    norm_keys = ["bn1", "bn2", "bn3", "shortcut"]

    def __init__(self, in_planes, planes, stride, conv_layer, conv_args,
                 act_layer, act_args, norm_layer, norm_args):
        super().__init__()

        self.regular_path = nn.Sequential(OrderedDict([
            # 1st layer
            ("conv1", conv_layer(in_planes, planes, kernel_size=1, bias=False,
                                 **conv_args["conv1x1_1"])),
            ("bn1", norm_layer(planes, **norm_args["bn1"])),
            ("act1", act_layer(planes, **act_args["act1"])),
            # 2nd layer
            ("conv2", conv_layer(planes, planes, stride=stride, kernel_size=3,
                                 padding=1, bias=False,
                                 **conv_args["conv3x3_2"])),
            ("bn2", norm_layer(planes, **norm_args["bn2"])),
            ("act2", act_layer(planes,
                               kernel_size=3,  # Deprecated, discarded by default
                               **act_args["act2"])),
            # 3rd layer
            ("conv3", conv_layer(planes, self.expansion * planes, kernel_size=1,
                                 bias=False, **conv_args["conv1x1_3"])),
            ("bn3", norm_layer(self.expansion * planes, **norm_args["bn3"])),
        ]))

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(OrderedDict([
                ("conv", conv_layer(in_planes, self.expansion * planes,
                                    kernel_size=1, stride=stride, bias=False,
                                    **conv_args["shortcut"])),
                ("bn", norm_layer(self.expansion * planes,
                                  **norm_args["shortcut"])),
            ]))
        else:
            self.shortcut = nn.Identity()

        self.post_activation = act_layer(self.expansion * planes,
                                         **act_args["act3"])

    def forward(self, x):
        out = self.regular_path(x)
        out += self.shortcut(x)
        out = self.post_activation(out)
        return out


# Number of blocks per group for different size Resnets.
cf_dict = {
    "18": (BasicBlock, [2, 2, 2, 2]),
    "34": (BasicBlock, [3, 4, 6, 3]),
    "50": (Bottleneck, [3, 4, 6, 3]),
    "101": (Bottleneck, [3, 4, 23, 3]),
    "152": (Bottleneck, [3, 8, 36, 3]),
}


def default_activation_layer(channels):
    """
    A wrapper function that takes the number of channels as an input. ReLU
    doesn't use this, but other activation functions may.
    """
    return nn.ReLU(inplace=True)


def discard_kernel_size(act_layer):
    """
    Used internally for compatibility.
    """
    def act_layer_discard_kernel_size(channels, kernel_size=0, **kwargs):
        return act_layer(channels, **kwargs)
    return act_layer_discard_kernel_size


class ResNet(nn.Module):
    """
    A customizable ResNet.

    It has two customization hooks: constructor functions and custom args.

    - Constructor functions let you configure the layer based on its args
      (e.g. in_channels, kernel_size, etc., and custom args)
    - Custom args let you configure based on the location in the network, or you
      can apply the same set of args to many layers.

    Example values:
      act_layer=my_relu_kwinners_activation_layer,
      act_args={"percent_on": 0.25}
      act_args={"stem": {},
                  "filters64": {},
                  "filters128": {"percent_on": 0.25},
                  ...}
      act_args={"stem": {},
                "filters64": [{}, {"act2": {"percent_on": 0.25}}, {}],
                ...}
      norm_args={"momentum": 0.11}
    """
    group_keys = ["filters64", "filters128", "filters256", "filters512"]

    def __init__(self,
                 depth=50,
                 num_classes=1000,
                 conv_layer=nn.Conv2d,
                 conv_args=None,
                 linear_layer=nn.Linear,
                 linear_args=None,
                 act_layer=default_activation_layer,
                 act_args=None,
                 norm_layer=nn.BatchNorm2d,
                 norm_args=None,
                 deprecated_compatibility_mode=False):
        """
        :param conv_layer:
            A conv2d layer that receives the arguments of a nn.Conv2d and custom
            conv_args
        :type conv_layer: callable

        :param conv_args:
            A dictionary specifying extra kwargs for the conv_layer, possibly
            assigning different args to each layer.
        :type conv_args: dict or None

        :param linear_layer:
            A linear layer that receives the arguments of a nn.Linear and custom
            linear_args
        :type conv_layer: callable

        :param linear_args:
            A dictionary specifying extra kwargs for the linear_layer, possibly
            assigning different args to each layer.
        :type conv_args: dict or None

        :param act_layer:
            An activation layer that receives the number of input channels and
            custom linear_args
        :type conv_layer: callable

        :param act_args:
            A dictionary specifying extra kwargs for the act_layer, possibly
            assigning different args to each layer.
        :type conv_args: dict or None

        :param norm_layer:
            A normalization layer that receives the arguments of nn.BatchNorm2d
            and custom norm_args
        :type conv_layer: callable

        :param norm_args:
            A dictionary specifying extra kwargs for the norm_layer, possibly
            assigning different args to each layer.
        :type conv_args: dict or None

        :param deprecated_compatibility_mode:
            Enables behavior required by SparseResNet
        :type deprecated_compatibility_mode: bool
        """

        super().__init__()

        assert str(depth) in cf_dict, "Resnet depth should be in {}".format(
            ",".join(cf_dict.keys()))
        block, num_blocks = cf_dict[str(depth)]

        conv_args = expand_args(conv_args, num_blocks, block.conv_keys)
        norm_args = expand_args(norm_args, num_blocks, block.norm_keys)
        act_args = expand_args(act_args, num_blocks, block.act_keys)
        linear_args = linear_args or {}

        if not deprecated_compatibility_mode:
            # Previous models expect to receive the kernel size in the
            # activation layer. Do this in the Bottleneck code, but discard it
            # by default.
            act_layer = discard_kernel_size(act_layer)

        features = [
            # stem
            ("stem", conv_layer(3, 64, kernel_size=7, stride=2,
                                padding=3, bias=False, **conv_args["stem"])),
            ("bn_stem", norm_layer(64, **norm_args["stem"])),
            ("act_stem", act_layer(64, **act_args["stem"])),
            ("pool_stem", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]

        # Track the previous out_channels during initialization.
        self.in_planes = 64
        features += [
            # groups 1 to 4
            ("group1", self._make_group(
                block, 64, num_blocks[0], stride=1,
                conv_layer=conv_layer, conv_args=conv_args["filters64"],
                act_layer=act_layer, act_args=act_args["filters64"],
                norm_layer=norm_layer, norm_args=norm_args["filters64"])),
            ("group2", self._make_group(
                block, 128, num_blocks[1], stride=2,
                conv_layer=conv_layer, conv_args=conv_args["filters128"],
                act_layer=act_layer, act_args=act_args["filters128"],
                norm_layer=norm_layer, norm_args=norm_args["filters128"])),
            ("group3", self._make_group(
                block, 256, num_blocks[2], stride=2,
                conv_layer=conv_layer, conv_args=conv_args["filters256"],
                act_layer=act_layer, act_args=act_args["filters256"],
                norm_layer=norm_layer, norm_args=norm_args["filters256"])),
            ("group4", self._make_group(
                block, 512, num_blocks[3], stride=2,
                conv_layer=conv_layer, conv_args=conv_args["filters512"],
                act_layer=act_layer, act_args=act_args["filters512"],
                norm_layer=norm_layer, norm_args=norm_args["filters512"])),
            ("avg_pool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", Flatten()),
        ]
        self.features = nn.Sequential(OrderedDict(features))
        del self.in_planes

        # last output layer
        self.classifier = linear_layer(
            512 * block.expansion,
            num_classes,
            **linear_args
        )

    def _make_group(self, block, planes, num_blocks, stride, conv_layer,
                    conv_args, act_layer, act_args, norm_layer, norm_args):
        strides = [stride] + [1] * (num_blocks - 1)

        assert len(conv_args) == num_blocks, (
            f"Length of args {len(conv_args)} should equal num of blocks "
            f"{num_blocks}")
        assert len(act_args) == num_blocks, (
            f"Length of args {len(act_args)} should equal num of blocks "
            f"{num_blocks}")
        assert len(norm_args) == num_blocks, (
            f"Length of args {len(norm_args)} should equal num of blocks "
            f"{num_blocks}")

        layers = []
        for stride, ca, aa, na in zip(strides, conv_args, act_args,
                                      norm_args):
            layers.append(block(self.in_planes, planes, stride=stride,
                                conv_layer=conv_layer, conv_args=ca,
                                act_layer=act_layer, act_args=aa,
                                norm_layer=norm_layer, norm_args=na))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


def expand_args(args, num_blocks_by_group, block_keys):
    """
    Converts a set of args into a layer-by-layer specification. The provided
    args may already be a layer-by-layer specification, or they may be a
    single dict that should be applied to every layer, or something in between.

    :param args:
        A dict specifying args
    :type args: dict or None

    :param num_blocks_by_group:
    :type num_blocks_by_group: list(int)

    :param block_keys:
        A list of keys that should exist for each block
    :type block_keys: list(string)

    :return:
        A dict specifying args for each group, block, and layer
    :rtype: dict
    """
    if args is None:
        args = {}

    top_level_keys = ["stem"] + ResNet.group_keys

    # If groups aren't specified, apply the config to every group.
    if any(k in args
           for k in top_level_keys):
        # Use the provided mapping, fill in any missing keys.
        args = {k: args.get(k, {})
                for k in top_level_keys}
    else:
        # Apply this dict to each group.
        args = {k: args
                for k in top_level_keys}

    for group, num_blocks in zip(ResNet.group_keys, num_blocks_by_group):
        group_args = args[group]

        # If blocks aren't specified, apply the config to every block.
        if not isinstance(group_args, (list, tuple)):
            group_args = [group_args] * num_blocks
        else:
            # Be careful not to mutate a data structure others are using.
            group_args = group_args.copy()

        # If layers aren't specified, apply the config to every layer.
        for i in range(len(group_args)):
            block_args = group_args[i]

            if any(k in block_args
                   for k in block_keys):
                # Use the provided mapping, fill in any missing keys.
                block_args = {k: block_args.get(k, {})
                              for k in block_keys}
            else:
                # Apply this dict to each layer.
                block_args = {k: block_args
                              for k in block_keys}
            group_args[i] = block_args
        args[group] = group_args

    return args


resnet18 = partial(ResNet, depth=18)
resnet34 = partial(ResNet, depth=34)
resnet50 = partial(ResNet, depth=50)
resnet101 = partial(ResNet, depth=101)
resnet152 = partial(ResNet, depth=152)
