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

from collections import namedtuple
from copy import deepcopy

import torch.nn as nn

import nupic.torch.modules as nupic_modules
from nupic.research.frameworks.pytorch.sparse_layer_params import (
    LayerParams,
    auto_sparse_activation_params,
    auto_sparse_conv_params,
)
from nupic.torch.modules import Flatten, KWinners2d

# Defines default convolutional params for different size conv layers
ConvParams = namedtuple("ConvParams", ["kernel_size", "padding"])
conv_types = {
    "1x1": ConvParams(kernel_size=1, padding=0),
    "3x3": ConvParams(kernel_size=3, padding=1),
    "5x5": ConvParams(kernel_size=5, padding=2),
    "7x7": ConvParams(kernel_size=7, padding=3),
}


def default_resnet_params(
    group_type,
    number_layers,
    layer_params_type=None,
    layer_params_kwargs=None,
):
    """
    Creates dictionary with default parameters.

    :param group_type: defines whether group is BasicBlock or Bottleneck.
    :param number_layers: number of layers to be assigned to each group.

    :returns dictionary with default parameters
    """
    layer_params_type = layer_params_type or LayerParams
    layer_params_kwargs = layer_params_kwargs or {}

    # Set layer params w/ activation.
    layer_params = layer_params_type(**layer_params_kwargs)

    # Set layer params w/o activation.
    noact_params_kwargs = deepcopy(layer_params_kwargs)
    noact_params_kwargs.pop("default_activation_params", None)
    noact_params_kwargs.pop("no_activation_params", None)
    noact_layer_params = layer_params_type(**noact_params_kwargs)

    # Validate layer_params
    assert isinstance(layer_params, LayerParams), \
        "Expected {} to sub-classed from LayerParams".format(layer_params)

    # Set layers params by group type.
    if group_type == BasicBlock:
        params = dict(
            conv3x3_1=layer_params, conv3x3_2=noact_layer_params, shortcut=layer_params
        )
    elif group_type == Bottleneck:
        params = dict(
            conv1x1_1=layer_params,
            conv3x3_2=layer_params,
            conv1x1_3=layer_params,
            shortcut=layer_params,
        )

    return dict(
        stem=layer_params,
        filters64=[params] * number_layers[0],
        filters128=[params] * number_layers[1],
        filters256=[params] * number_layers[2],
        filters512=[params] * number_layers[3],
        linear=noact_layer_params,
    )


def linear_layer(input_size, output_size, layer_params, sparse_weights_type):
    """Basic linear layer, which accepts different sparse layer types."""
    layer = nn.Linear(input_size, output_size)

    # Compute params for sparse-weights module.
    if layer_params is not None:
        weight_params = layer_params.get_linear_params(
            input_size,
            output_size,
        )
    else:
        weight_params = None

    # Initialize sparse-weights module as specified.
    if weight_params is not None:
        return sparse_weights_type(layer, **weight_params)
    else:
        return layer


def conv_layer(
    conv_type,
    in_planes,
    out_planes,
    layer_params,
    sparse_weights_type,
    stride=1,
    bias=False,
):
    """Basic conv layer, which accepts different sparse layer types."""
    kernel_size, padding = conv_types[conv_type]
    layer = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    # Compute params for sparse-weights module.
    if layer_params is not None:
        weight_params = layer_params.get_conv_params(
            in_planes,
            out_planes,
            kernel_size
        )
    else:
        weight_params = None

    # Initialize sparse-weights module as specified.
    if weight_params is not None:
        return sparse_weights_type(layer, **weight_params)
    else:
        return layer


def activation_layer(
    out,
    layer_params,
    kernel_size=0
):
    """Basic activation layer.
    Defaults to ReLU if `activation_params` are evaluated from `layer_params`.
    Otherwise KWinners is used."""

    # Compute layer_params for kwinners activation module.
    if layer_params is not None:
        activation_params = layer_params.get_activation_params(0, out, kernel_size)
    else:
        activation_params = None

    # Initialize kwinners module as specified.
    if activation_params is not None:
        return nn.Sequential(
            KWinners2d(
                out,
                **activation_params
            ),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.ReLU(inplace=True)


class BasicBlock(nn.Module):
    """Default block for ResNets with < 50 layers."""

    expansion = 1

    def __init__(self, in_planes, planes, sparse_weights_type, layer_params, stride=1):
        super(BasicBlock, self).__init__()

        self.regular_path = nn.Sequential(
            conv_layer(
                "3x3",
                in_planes,
                planes,
                layer_params["conv3x3_1"],
                sparse_weights_type=sparse_weights_type,
                stride=stride,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, layer_params["conv3x3_1"]),
            conv_layer(
                "3x3",
                planes,
                planes,
                layer_params["conv3x3_2"],
                sparse_weights_type=sparse_weights_type,
            ),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    "1x1",
                    in_planes,
                    planes,
                    layer_params["shortcut"],
                    sparse_weights_type=sparse_weights_type,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes),
            )

        self.post_activation = activation_layer(planes, layer_params["shortcut"])

    def forward(self, x):
        out = self.regular_path(x)
        out += self.shortcut(x)
        out = self.post_activation(out)
        return out


class Bottleneck(nn.Module):
    """Default block for ResNets with >= 50 layers."""

    expansion = 4

    def __init__(self, in_planes, planes, sparse_weights_type, layer_params, stride=1):
        super(Bottleneck, self).__init__()
        self.regular_path = nn.Sequential(
            # 1st layer
            conv_layer(
                "1x1",
                in_planes,
                planes,
                layer_params["conv1x1_1"],
                sparse_weights_type=sparse_weights_type,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, layer_params["conv1x1_1"], kernel_size=1),
            # 2nd layer
            conv_layer(
                "3x3",
                planes,
                planes,
                layer_params["conv3x3_2"],
                sparse_weights_type=sparse_weights_type,
                stride=stride,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, layer_params["conv3x3_2"], kernel_size=3),
            # 3rd layer
            conv_layer(
                "1x1",
                planes,
                self.expansion * planes,
                layer_params["conv1x1_3"],
                sparse_weights_type=sparse_weights_type,
            ),
            nn.BatchNorm2d(self.expansion * planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    "1x1",
                    in_planes,
                    self.expansion * planes,
                    layer_params["shortcut"],
                    sparse_weights_type=sparse_weights_type,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.post_activation = activation_layer(
            self.expansion * planes, layer_params["shortcut"], kernel_size=1
        )

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

# URLs to access pretrained models
model_urls = {
    18: "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    34: "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    50: "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    101: "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    152: "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class ResNet(nn.Module):
    """Based of torchvision Resnet @
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"""

    def __init__(self, config=None):
        super(ResNet, self).__init__()

        # update config
        defaults = dict(
            depth=50,
            num_classes=1000,
            linear_sparse_weights_type="SparseWeights",
            conv_sparse_weights_type="SparseWeights2d",
            defaults_sparse=False,
            layer_params_type=None,  # Sub-classed from `LayerParams`.
            layer_params_kwargs=None,  # to be passed to layer_params_type.
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        if isinstance(self.linear_sparse_weights_type, str):
            self.linear_sparse_weights_type = getattr(
                nupic_modules, self.linear_sparse_weights_type)
        if isinstance(self.conv_sparse_weights_type, str):
            self.conv_sparse_weights_type = getattr(
                nupic_modules, self.conv_sparse_weights_type)

        if self.defaults_sparse:
            self.layer_params_kwargs = self.layer_params_kwargs or {}
            self.layer_params_kwargs.setdefault(
                "conv_params_func", auto_sparse_conv_params)
            self.layer_params_kwargs.setdefault(
                "activation_params_func", auto_sparse_activation_params)

        if not hasattr(self, "sparse_params"):
            self.sparse_params = default_resnet_params(
                *cf_dict[str(self.depth)],
                layer_params_type=self.layer_params_type,
                layer_params_kwargs=self.layer_params_kwargs,
            )

        self.in_planes = 64

        block, num_blocks = self._config_layers()

        self.features = nn.Sequential(
            # stem
            conv_layer(
                "7x7",
                3,
                64,
                self.sparse_params["stem"],
                sparse_weights_type=self.conv_sparse_weights_type,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            activation_layer(64, self.sparse_params["stem"], kernel_size=7),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # groups 1 to 4
            self._make_group(
                block, 64, num_blocks[0], self.sparse_params["filters64"], stride=1
            ),
            self._make_group(
                block, 128, num_blocks[1], self.sparse_params["filters128"], stride=2
            ),
            self._make_group(
                block, 256, num_blocks[2], self.sparse_params["filters256"], stride=2
            ),
            self._make_group(
                block, 512, num_blocks[3], self.sparse_params["filters512"], stride=2
            ),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

        # last output layer
        self.classifier = linear_layer(
            512 * block.expansion,
            self.num_classes,
            self.sparse_params["linear"],
            self.linear_sparse_weights_type,
        )

    def _config_layers(self):
        depth_lst = [18, 34, 50, 101, 152]
        assert (
            self.depth in depth_lst
        ), "Error : Resnet depth should be either 18, 34, 50, 101, 152"

        return cf_dict[str(self.depth)]

    def _make_group(self, block, planes, num_blocks, sparse_params, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        # allows sparse params to be defined per group
        if type(sparse_params) == dict:
            sparse_params = [sparse_params] * num_blocks

        assert (
            len(sparse_params) == num_blocks
        ), "Length of sparse params {:d} should equal num of blocks{:d}".format(
            len(sparse_params), num_blocks
        )

        for layer_params, stride in zip(sparse_params, strides):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    layer_params=layer_params,
                    sparse_weights_type=self.conv_sparse_weights_type,
                    stride=stride,
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


# convenience classes
def build_resnet(depth, config=None):
    config = config or {}
    config["depth"] = depth
    return ResNet(config)


def resnet18(config=None):
    return build_resnet(18, config)


def resnet34(config=None):
    return build_resnet(34, config)


def resnet50(config=None):
    return build_resnet(50, config)


def resnet101(config=None):
    return build_resnet(101, config)


def resnet152(config=None):
    return build_resnet(152, config)
