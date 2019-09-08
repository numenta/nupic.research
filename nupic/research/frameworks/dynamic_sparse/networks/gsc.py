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
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import torch
from torch import nn

from nupic.torch.models.sparse_cnn import MNISTSparseCNN
from nupic.torch.modules import Flatten, KWinners, KWinners2d

from .layers import DSConv2d, RandDSConv2d, SparseConv2d
from .main import VGG19

# redefine Flatten
# class Lambda(nn.Module):
#     def __init__(self, func:LambdaFunc):
#         super().__init__()
#         self.func = func

#     def forward(self, x):
#         return self.func(x)

# def Flatten():
#     return Lambda(lambda x: x.view((x.size(0), -1)))


class GSCHeb(nn.Module):
    """LeNet like CNN used for GSC in how so dense paper."""

    def __init__(self, config=None):
        super(GSCHeb, self).__init__()

        defaults = dict(
            input_size=1024,
            num_classes=12,
            boost_strength=1.5,
            boost_strength_factor=0.9,
            k_inference_factor=1.5,
            duty_cycle_period=1000,
            use_kwinners=True,
            hidden_neurons_fc=1000,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        if self.model == "DSNNMixedHeb":
            self.hebbian_learning = True
        else:
            self.hebbian_learning = False

        # hidden layers
        conv_layers = [
            *self._conv_block(1, 64, percent_on=0.095),  # 28x28 -> 14x14
            *self._conv_block(64, 64, percent_on=0.125),  # 10x10 -> 5x5
        ]
        linear_layers = [
            Flatten(),
            # *self._linear_block(1600, 1500, percent_on= 0.067),
            *self._linear_block(1600, self.hidden_neurons_fc, percent_on=0.1),
            nn.Linear(self.hidden_neurons_fc, self.num_classes),
        ]

        # classifier (*redundancy on layers to facilitate traversing)
        self.layers = conv_layers + linear_layers
        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(*linear_layers)

        # track correlations
        self.correlations = []

    def _conv_block(self, fin, fout, percent_on=0.1):
        block = [
            nn.Conv2d(fin, fout, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(fout, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._activation_func(fout, percent_on),
        ]
        # if not self.use_kwinners:
        #     block.append(nn.Dropout(p=0.5))
        return block

    def _linear_block(self, fin, fout, percent_on=0.1):
        block = [
            nn.Linear(fin, fout),
            nn.BatchNorm1d(fout, affine=False),
            self._activation_func(fout, percent_on, twod=False),
        ]
        # if not self.use_kwinners:
        #     block.append(nn.Dropout(p=0.5))
        return block

    def _activation_func(self, fout, percent_on, twod=True):
        if self.use_kwinners:
            if twod:
                activation_func = KWinners2d
            else:
                activation_func = KWinners
            return activation_func(
                fout,
                percent_on=percent_on,
                boost_strength=self.boost_strength,
                boost_strength_factor=self.boost_strength_factor,
                k_inference_factor=self.k_inference_factor,
                duty_cycle_period=self.duty_cycle_period,
            )
        else:
            return nn.ReLU()

    def _has_activation(self, idx, layer):
        return idx == len(self.layers) - 1 or isinstance(layer, KWinners)

    def forward(self, x):
        """A faster and approximate way to track correlations"""
        # Forward pass through conv layers
        for layer in self.features:
            x = layer(x)

        # Forward pass through linear layers
        idx_activation = 0
        for layer in self.classifier:
            # do the forward calculation normally
            x = layer(x)
            if self.hebbian_learning:
                if isinstance(layer, Flatten):
                    prev_act = (x > 0).detach().float()
                if isinstance(layer, KWinners):
                    n_samples = x.shape[0]
                    with torch.no_grad():
                        curr_act = (x > 0).detach().float()
                        # add outer product to the correlations, per sample
                        for s in range(n_samples):
                            outer = torch.ger(prev_act[s], curr_act[s])
                            if idx_activation + 1 > len(self.correlations):
                                self.correlations.append(outer)
                            else:
                                self.correlations[idx_activation] += outer
                        # reassigning to the next
                        prev_act = curr_act
                        # move to next activation
                        idx_activation += 1

        return x


# make a conv heb just by replacing the conv layers by special DSNN Conv layers
def gsc_conv_heb(config):
    # return make_dscnn(models.vgg19_bn(config), config)
    return make_dscnn(GSCHeb(config), config)


def gsc_conv_only_heb(config):
    network = make_dscnn(GSCHeb(config), config)

    # replace the forward function to not apply regular convolution
    def forward(self, x):
        return self.classifier(self.features(x))

    network.forward = forward

    return network


# function that makes the switch
# why function inside other functions -> make it into a class?


def make_dscnn(net, config=None):

    config = config or {}

    named_convs = [
        (name, layer)
        for name, layer in net.named_modules()
        if isinstance(layer, torch.nn.Conv2d)
    ]
    num_convs = len(named_convs)

    def tolist(param):
        if isinstance(param, list):
            return param
        else:
            return [param] * num_convs

    def get_conv_type(prune_method):
        if prune_method == "random":
            return RandDSConv2d
        elif prune_method == "static":
            return SparseConv2d
        elif prune_method == "dynamic":
            return DSConv2d

    # Get DSConv2d params from config.
    prune_methods = tolist(config.get("prune_methods", "dynamic"))
    assert (
        len(prune_methods) == num_convs
    ), "Not enough prune_methods specified in config. Expected {}, got {}".format(
        num_convs, prune_methods
    )

    possible_args = {
        "dynamic": [
            "hebbian_prune_frac",
            "weight_prune_frac",
            "sparsity",
            "prune_dims",
            "update_nsteps",
        ],
        "random": [
            "hebbian_prune_frac",
            "weight_prune_frac",
            "sparsity",
            "prune_dims",
            "update_nsteps",
        ],
        "static": ["sparsity"],
        None: [],
    }
    kwargs_s = []
    for c_i in range(num_convs):
        layer_args = {}
        prune_method = prune_methods[c_i]
        for arg in possible_args[prune_method]:
            if arg in config:
                layer_args[arg] = tolist(config.get(arg))[c_i]
        kwargs_s.append(layer_args)

    assert (
        len((kwargs_s)) == len(named_convs) == len(prune_methods)
    ), "Sizes do not match"

    # OLD VERSION

    # for prune_meth, kwargs, (name, conv) in zip(prune_methods, kwargs_s, named_convs):

    #     NewConv = get_conv_type(prune_meth)
    #     if NewConv is None:
    #         continue

    #     setattr(net, name, NewConv(
    #         in_channels=conv.in_channels,
    #         out_channels=conv.out_channels,
    #         kernel_size=conv.kernel_size,
    #         stride=conv.stride,
    #         padding=conv.padding,
    #         padding_mode=conv.padding_mode,
    #         dilation=conv.dilation,
    #         groups=conv.groups,
    #         bias=(conv.bias is not None),
    #         **kwargs,
    #     ))

    # NEW VERSION

    new_features = []
    idx = 0  # iterate through args
    # only start procedure if there at least one layer which needs to be pruned
    if len(prune_methods):
        # replace all conv layers if required
        for layer in net.features:
            if isinstance(layer, nn.Conv2d):
                # only replace layer if one of the expected types
                prune_method = prune_methods[idx]
                if prune_method in ["static", "random", "dynamic"]:
                    kwargs = kwargs_s[idx]
                    conv_type = get_conv_type(prune_method)
                    new_features.append(
                        conv_type(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                            padding_mode=layer.padding_mode,
                            dilation=layer.dilation,
                            groups=layer.groups,
                            bias=(layer.bias is not None),
                            **kwargs,
                        )
                    )
                # else, do nothing, append conv
                else:
                    new_features.append(layer)
                idx += 1
            else:
                # do nothing, append regular layer
                new_features.append(layer)

    net.features = nn.Sequential(*new_features)

    return net


def vgg19_dscnn(config):

    net = VGG19(config)
    net = make_dscnn(net)
    return net


def mnist_sparse_cnn(config):

    net = MNISTSparseCNN()
    return net


def mnist_sparse_dscnn(config):

    net = MNISTSparseCNN()
    net = make_dscnn(net, config)
    return net
