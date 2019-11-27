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

import torch.nn as nn

import nupic.torch.modules as nupic_modules
from nupic.torch.modules import Flatten, KWinners2d

# Defines default convolutional params for different size conv layers
ConvParams = namedtuple("ConvParams", ["kernel_size", "padding"])
conv_types = {
    "1x1": ConvParams(kernel_size=1, padding=0),
    "3x3": ConvParams(kernel_size=3, padding=1),
    "5x5": ConvParams(kernel_size=5, padding=2),
    "7x7": ConvParams(kernel_size=7, padding=3),
}

# Defines default sparse params for regular layers with activations
LayerParams = namedtuple(
    "LayerParams",
    [
        "percent_on",
        "boost_strength",
        "boost_strength_factor",
        "k_inference_factor",
        "local",
        "weights_density",
    ],
)
LayerParams.__new__.__defaults__ = (1.0, 1.4, 0.7, 1.0, True, 1.0)

# Defines default sparse params for layers without activations
NoactLayerParams = namedtuple("NoactLayerParams", ["weights_density"])
NoactLayerParams.__new__.__defaults__ = (1.0,)


def default_sparse_params(group_type, number_layers, sparse=False):
    """Creates dictionary with default parameters.
    If sparse_params is passed to the model, default params are not used.

    :param group_type: defines whether group is BasicBlock or Bottleneck
    :param number_layers: number of layers to be assigned to each group

    :returns dictionary with default parameters
    """
    if sparse:
        layer_params = LayerParams(0.25, 1.4, 0.7, 1.0, True, 0.5)
        noact_layer_params = NoactLayerParams(0.5)
    else:   
        layer_params = LayerParams()
        noact_layer_params = NoactLayerParams()

    if group_type == BasicBlock:
        params = dict(
            conv3x3_1=layer_params,
            conv3x3_2=noact_layer_params,
            shortcut=layer_params,
        )
    elif group_type == Bottleneck:
        params = dict(
            conv1x1_1=layer_params,
            conv3x3_2=layer_params,
            conv1x1_3=noact_layer_params,
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


def linear_layer(input_size, output_size, weights_density, sparse_weights_type):
    """Basic linear layer, which accepts different sparse layer types."""
    layer = nn.Linear(input_size, output_size)

    # adds sparsity to last linear layer
    if weights_density < 1.0:
        sparse_weights_type = getattr(nupic_modules, sparse_weights_type)
        return sparse_weights_type(layer, weights_density)
    else:
        return layer


def conv_layer(
    conv_type,
    in_planes,
    out_planes,
    weights_density,
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
    if weights_density < 1.0:
        sparse_weights_type = getattr(nupic_modules, sparse_weights_type)
        return sparse_weights_type(layer, weights_density)
    else:
        return layer


def activation_layer(
    out,
    percent_on,
    boost_strength,
    boost_strength_factor,
    k_inference_factor,
    local,
    *args,
):
    """Basic activation layer.
    Defaults to ReLU if percent_on is < 0.5. Otherwise KWinners is used."""
    if percent_on >= 0.5:
        return nn.ReLU(inplace=True)
    else:
        return KWinners2d(
            out,
            percent_on=percent_on,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            k_inference_factor=k_inference_factor,
            local=local
        )


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
                layer_params["conv3x3_1"].weights_density,
                sparse_weights_type=sparse_weights_type,
                stride=stride,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, *layer_params["conv3x3_1"]),
            conv_layer(
                "3x3",
                planes,
                planes,
                layer_params["conv3x3_2"].weights_density,
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
                    layer_params["shortcut"].weights_density,
                    sparse_weights_type=sparse_weights_type,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes),
            )

        self.post_activation = activation_layer(planes, *layer_params["shortcut"])

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
                layer_params["conv1x1_1"].weights_density,
                sparse_weights_type=sparse_weights_type,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, *layer_params["conv1x1_1"]),
            # 2nd layer
            conv_layer(
                "3x3",
                planes,
                planes,
                layer_params["conv3x3_2"].weights_density,
                sparse_weights_type=sparse_weights_type,
                stride=stride,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, *layer_params["conv3x3_2"]),
            # 3rd layer
            conv_layer(
                "1x1",
                planes,
                self.expansion * planes,
                layer_params["conv1x1_3"].weights_density,
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
                    layer_params["shortcut"].weights_density,
                    sparse_weights_type=sparse_weights_type,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

        self.post_activation = activation_layer(
            self.expansion * planes, *layer_params["shortcut"]
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
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)

        if not hasattr(self, "sparse_params"):
            self.sparse_params = default_sparse_params(*cf_dict[str(self.depth)], sparse=self.defaults_sparse)

        self.in_planes = 64

        block, num_blocks = self._config_layers()

        self.features = nn.Sequential(
            # stem
            conv_layer(
                "7x7",
                3,
                64,
                self.sparse_params["stem"].weights_density,
                sparse_weights_type=self.conv_sparse_weights_type,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            activation_layer(64, *self.sparse_params["stem"]),
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
            self.sparse_params["linear"].weights_density,
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