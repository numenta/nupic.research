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

import torch
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
        "percent_on_k_winner",
        "boost_strength",
        "boost_strength_factor",
        "k_inference_factor",
        "weights_density",
    ],
    defaults=[0.25, 1.4, 0.7, 1.0, 0.5],
)

# Defines default sparse params for layers without activations
NoactLayerParams = namedtuple("NoactLayerParams", ["weights_density"], defaults=[0.5])


def default_sparse_params(group_type, number_layers):
    """Creates dictionary with default parameters.
    If sparse_params is passed to the model, default params are not used.

    :param group_type: defines whether group is BasicBlock or Bottleneck
    :param number_layers: number of layers to be assigned to each group

    :returns dictionary with default parameters
    """
    if group_type == BasicBlock:
        params = dict(
            conv3x3_1=LayerParams(),
            conv3x3_2=NoactLayerParams(),
            shortcut=LayerParams(),
        )
    elif group_type == Bottleneck:
        params = dict(
            conv1x1_1=LayerParams(),
            conv3x3_2=LayerParams(),
            conv1x1_3=NoactLayerParams(),
            shortcut=LayerParams(),
        )

    return dict(
        stem=LayerParams(),
        filters64=[params] * number_layers[0],
        filters128=[params] * number_layers[1],
        filters256=[params] * number_layers[2],
        filters512=[params] * number_layers[3],
        linear=NoactLayerParams(),
    )


def linear_layer(input_size, output_size, weights_density, sparse_layer_type):
    """Basic linear layer, which accepts different sparse layer types."""
    layer = nn.Linear(input_size, output_size)

    # adds sparsity to last linear layer
    if weights_density < 1.0:
        sparse_layer_type = getattr(nupic_modules, sparse_layer_type)
        return sparse_layer_type(layer, weights_density)
    else:
        return layer


def conv_layer(
    conv_type,
    in_planes,
    out_planes,
    weights_density,
    sparse_layer_type,
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
        sparse_layer_type = getattr(nupic_modules, sparse_layer_type)
        return sparse_layer_type(layer, weights_density)
    else:
        return layer


def activation_layer(
    out,
    percent_on_k_winner,
    boost_strength,
    boost_strength_factor,
    k_inference_factor,
    *args,
):
    """Basic activation layer.
    Defaults to ReLU if percent_on is < 0.5. Otherwise KWinners is used."""
    if percent_on_k_winner >= 0.5:
        return nn.ReLU()
    else:
        return KWinners2d(
            out,
            percent_on=percent_on_k_winner,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            k_inference_factor=k_inference_factor,
        )


class BasicBlock(nn.Module):
    """Default block for ResNets with < 50 layers."""

    expansion = 1

    def __init__(self, in_planes, planes, sparse_layer_type, layer_params, stride=1):
        super(BasicBlock, self).__init__()

        self.regular_path = nn.Sequential(
            conv_layer(
                "3x3",
                in_planes,
                planes,
                layer_params["conv3x3_1"].weights_density,
                sparse_layer_type=sparse_layer_type,
                stride=stride,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, *layer_params["conv3x3_1"]),
            conv_layer(
                "3x3",
                planes,
                planes,
                layer_params["conv3x3_2"].weights_density,
                sparse_layer_type=sparse_layer_type,
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
                    sparse_layer_type=sparse_layer_type,
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

    def __init__(self, in_planes, planes, sparse_layer_type, layer_params, stride=1):
        super(Bottleneck, self).__init__()
        self.regular_path = nn.Sequential(
            # 1st layer
            conv_layer(
                "1x1",
                in_planes,
                planes,
                layer_params["conv1x1_1"].weights_density,
                sparse_layer_type=sparse_layer_type,
            ),
            nn.BatchNorm2d(planes),
            activation_layer(planes, *layer_params["conv1x1_1"]),
            # 2nd layer
            conv_layer(
                "3x3",
                planes,
                planes,
                layer_params["conv3x3_2"].weights_density,
                sparse_layer_type=sparse_layer_type,
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
                sparse_layer_type=sparse_layer_type,
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
                    sparse_layer_type=sparse_layer_type,
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
            num_classes=10,
            sparse_linear_layer_type="SparseWeights",
            sparse_conv_layer_type="SparseWeights2d",
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)

        if not hasattr(self, "sparse_params"):
            self.sparse_params = default_sparse_params(*cf_dict[str(self.depth)])

        self.in_planes = 64

        block, num_blocks = self._config_layers()

        self.features = nn.Sequential(
            # stem
            conv_layer(
                "7x7",
                3,
                64,
                self.sparse_params["stem"].weights_density,
                sparse_layer_type=self.sparse_conv_layer_type,
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
            self.sparse_linear_layer_type,
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
                    sparse_layer_type=self.sparse_conv_layer_type,
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


# base tests
if __name__ == "__main__":

    from torch.autograd import Variable

    # ----- Regular resnet, not customized

    # regular resnet, not customized
    net = ResNet(config=dict(depth=50, num_classes=10))
    y = net(Variable(torch.randn(2, 3, 32, 32)))
    print("ResNet50 with default parameters: ok")

    # ----- Resnets customized per group

    custom_sparse_params = dict(
        stem=LayerParams(),
        filters64=dict(
            conv1x1_1=LayerParams(
                percent_on_k_winner=0.3,
                boost_strength=1.2,
                boost_strength_factor=1.0,
                weights_density=0.3,
            ),
            conv3x3_2=LayerParams(
                percent_on_k_winner=0.1,
                boost_strength=1.2,
                boost_strength_factor=1.0,
                weights_density=0.1,
            ),
            conv1x1_3=NoactLayerParams(weights_density=0.1),
            shortcut=LayerParams(
                percent_on_k_winner=0.4,
                boost_strength=1.2,
                boost_strength_factor=1.0,
                weights_density=0.4,
            ),
        ),
        filters128=dict(
            conv1x1_1=LayerParams(),
            conv3x3_2=LayerParams(),
            conv1x1_3=NoactLayerParams(),
            shortcut=LayerParams(),
        ),
        filters256=dict(
            conv1x1_1=LayerParams(),
            conv3x3_2=LayerParams(),
            conv1x1_3=NoactLayerParams(),
            shortcut=LayerParams(),
        ),
        filters512=dict(
            conv1x1_1=LayerParams(),
            conv3x3_2=LayerParams(),
            conv1x1_3=NoactLayerParams(),
            shortcut=LayerParams(),
        ),
        linear=NoactLayerParams(weights_density=0.5),
    )

    net = ResNet(
        config=dict(depth=50, num_classes=10, sparse_params=custom_sparse_params)
    )
    y = net(Variable(torch.randn(2, 3, 32, 32)))
    print("ResNet50 customized per group: ok")

    # ----- Fully customized resnets

    custom_sparse_params = dict(
        stem=LayerParams(),
        filters64=[  # 3 blocks
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
        ],
        filters128=[  # 4 blocks
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
        ],
        filters256=[  # 6 blocks
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
        ],
        filters512=[  # 3 blocks
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
        ],
        linear=NoactLayerParams(),
    )

    net = ResNet(
        config=dict(depth=50, num_classes=10, sparse_params=custom_sparse_params)
    )
    y = net(Variable(torch.randn(2, 3, 32, 32)))
    print("ResNet50 fully customized: ok")

    print(net)

    # ----- Test different size resnets

    # larger resnet
    net = resnet101()
    y = net(Variable(torch.randn(2, 3, 32, 32)))

    # smaller resnet
    net = resnet18()
    y = net(Variable(torch.randn(2, 3, 32, 32)))
