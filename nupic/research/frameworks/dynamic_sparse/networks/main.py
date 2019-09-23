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

import math

import torch
from torch import nn
from torchvision import models

from nupic.torch.modules import Flatten, KWinners, KWinners2d

from .layers import DSConv2d, DSLinear
from .utils import (
    get_dynamic_sparse_modules,
    make_dsnn,
    replace_sparse_weights,
    squash_layers,
    swap_layers,
)


class VGG19(nn.Module):
    def __init__(self, config=None):
        super(VGG19, self).__init__()

        defaults = dict(
            device="gpu",
            num_classes=10,
            init_weights=True,
            kwinners=True,
            boost_strength=1.5,
            boost_strength_factor=0.85,
            percent_on=0.3,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        # define if kwinners or regular network
        if self.kwinners:
            self.pool_func = lambda: nn.AvgPool2d(kernel_size=2, stride=2)
            self.nonlinear_func = self._kwinners
        else:
            self.pool_func = lambda: nn.MaxPool2d(kernel_size=2, stride=2)
            self.nonlinear_func = lambda fout: nn.ReLU()

        # initialize network
        layers = [
            *self._conv_block(3, 64),
            *self._conv_block(64, 64, pool=True),  # 16x16
            *self._conv_block(64, 128),
            *self._conv_block(128, 128, pool=True),  # 8x8
            *self._conv_block(128, 256),
            *self._conv_block(256, 256),
            *self._conv_block(256, 256),
            *self._conv_block(256, 256, pool=True),  # 4x4
            *self._conv_block(256, 512),
            *self._conv_block(512, 512),
            *self._conv_block(512, 512),
            *self._conv_block(512, 512, pool=True),  # 2x2
            *self._conv_block(512, 512),
            *self._conv_block(512, 512),
            *self._conv_block(512, 512),
            *self._conv_block(512, 512, pool=True),  # 1x1
        ]
        layers.append(Flatten())
        layers.append(nn.Linear(512, self.num_classes))
        self.classifier = nn.Sequential(*layers)

        if self.init_weights:
            self.initialize_weights(self.classifier)

    def _kwinners(self, fout):
        return KWinners2d(
            channels=fout,
            percent_on=self.percent_on,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
        )

    def _conv_block(self, fin, fout, pool=False):
        block = [
            nn.Conv2d(fin, fout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fout),
            self.nonlinear_func(fout),
        ]
        if pool:
            block.append(self.pool_func())
        return block

    def forward(self, x):
        return self.classifier(x)

    @classmethod
    def initialize_weights(cls, net):
        """
        Initializes weights for any network where the initialization-scheme
        follows that of the VGG-19 network in the "How can we be so Dense"
        paper.
        """
        for m in net.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG19HebDepreciated(nn.Module):
    def __init__(self, config=None):
        super(VGG19HebDepreciated, self).__init__()

        defaults = dict(
            device="gpu",
            input_size=784,
            num_classes=10,
            hidden_sizes=[4000, 1000, 4000],
            batch_norm=False,
            dropout=0.3,
            bias=False,
            init_weights=True,
            kwinners=False,
            percent_on=0.3,
            boost_strength=1.4,
            boost_strength_factor=0.7,
            hebbian_learning=True,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        # define if kwinners or regular network
        if self.kwinners:
            self.pool_func = lambda: nn.AvgPool2d(kernel_size=2, stride=2)
            self.nonlinear_func = self._kwinners
        else:
            self.pool_func = lambda: nn.MaxPool2d(kernel_size=2, stride=2)
            self.nonlinear_func = lambda fout: nn.ReLU()

        # initialize network
        layers = [
            *self._conv_block(3, 64, pool=True),  # 16x16
            *self._conv_block(64, 64, pool=True),  # 8x8
            *self._conv_block(64, 128, pool=True),  # 4x4
            *self._conv_block(128, 256, pool=True),  # 2x2
            *self._conv_block(256, 512, pool=True),  # 1x1
        ]
        layers.append(Flatten())
        layers.append(nn.Linear(512, self.num_classes))
        self.classifier = nn.Sequential(*layers)

        # track the activations
        # should reset at the end of each round, done in the model
        self.correlations = []

        if self.init_weights:
            self._initialize_weights()

    def _kwinners(self, fout):
        return KWinners2d(
            channels=fout,
            percent_on=self.percent_on,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
        )

    def _conv_block(self, fin, fout, pool=False):
        block = [
            nn.Conv2d(fin, fout, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fout),
            self.nonlinear_func(fout),
        ]
        if pool:
            block.append(self.pool_func())
        return block

        # track correlations
        self.correlations = []

    def _has_activation(self, idx, layer):
        return (
            idx == len(self.layers) - 1
            or isinstance(layer, nn.ReLU)
            or isinstance(layer, KWinners)
        )

    def forward(self, x):
        """Same idea as in MLP
        TODO: only get the receptive field - need to find the reverse
        heuristic to get the right activations,considering there is likely
        padding as well

        Sample implementation, not tested
        """
        x = x.view(-1, self.input_size)  # resiaze if needed, eg mnist
        prev_act = (x > 0).detach().float()
        idx_activation = 0
        for idx_layer, layer in enumerate(self.layers):
            # do the forward calculation normally
            x = layer(x)
            if self.hebbian_learning:
                n_samples = x.shape[0]
                if self._has_activation(idx_layer, layer):
                    with torch.no_grad():
                        curr_act = (x > 0).detach().float()
                        correlations = torch.zeros(prev_act.shape)
                        # loop across samples
                        for s in range(n_samples):
                            # loop across channels
                            for c in range(curr_act.shape[1]):
                                for x in range(curr_act.shape[2]):
                                    for y in range(curr_act.shape[3]):
                                        # TODO: only get the receptive field
                                        x_range, y_range = None, None
                                        joint_act = (
                                            prev_act[s, x_range, y_range, :]
                                            * curr_act[s, x, y, c]
                                        )
                                        correlations[
                                            s, x_range, y_range, :
                                        ] += joint_act

                        # at the end, sum over all samples
                        correlations_total = torch.sum(correlations, dim=0)
                        # update corre
                        if idx_activation + 1 > len(self.correlations):
                            self.correlations.append(correlations_total)
                        else:
                            self.correlations[idx_activation] += correlations_total

                        # reassing to the next
                        prev_act = curr_act
                        # move to next activation
                        idx_activation += 1

        return x

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class MLPDeprecated(nn.Module):
    """
    Simple 3 hidden layers + output MLP, similar to one used in SET Paper.
    """

    def __init__(self, config=None):
        super().__init__()

        defaults = dict(
            input_size=784,
            num_classes=10,
            hidden_sizes=[4000, 1000, 4000],
            batch_norm=False,
            dropout=0.3,
            bias=False,
            init_weights=True,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        layers = []
        layers.extend(
            self.linear_block(
                self.input_size,
                self.hidden_sizes[0],
                bn=self.batch_norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        )
        layers.extend(
            self.linear_block(
                self.hidden_sizes[0],
                self.hidden_sizes[1],
                bn=self.batch_norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        )
        layers.extend(
            self.linear_block(
                self.hidden_sizes[1],
                self.hidden_sizes[2],
                bn=self.batch_norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        )

        # output layer
        layers.append(nn.Linear(self.hidden_sizes[2], self.num_classes, bias=self.bias))
        self.classifier = nn.Sequential(*layers)

        if self.init_weights:
            self._initialize_weights(self.bias)

    @staticmethod
    def linear_block(a, b, bn=False, dropout=False, bias=True):
        block = [nn.Linear(a, b, bias=bias), nn.ReLU()]
        if bn:
            block.append(nn.BatchNorm1d(b))
        if dropout:
            block.append(nn.Dropout(p=dropout))

        return block

    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size))

    def _initialize_weights(self, bias):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if bias:
                    nn.init.constant_(m.bias, 0)


class MLPHebDeprecated(nn.Module):
    """
    Simple 3 hidden layers + output MLPHeb, similar to one used in SET Paper.
    """

    def __init__(self, config=None):
        super().__init__()

        defaults = dict(
            input_size=784,
            num_classes=10,
            hidden_sizes=[1000, 1000, 1000],
            batch_norm=False,
            dropout=False,
            bias=False,
            init_weights=True,
            hebbian_learning=False,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        if self.kwinners:
            self.activation_func = self._kwinners
        else:
            self.activation_func = lambda fout: nn.ReLU()

        # hidden layers
        layers = [
            *self._linear_block(self.input_size, self.hidden_sizes[0]),
            *self._linear_block(self.hidden_sizes[0], self.hidden_sizes[1]),
            *self._linear_block(self.hidden_sizes[1], self.hidden_sizes[2]),
        ]
        # output layer
        layers.append(nn.Linear(self.hidden_sizes[2], self.num_classes, bias=self.bias))

        # classifier (*redundancy on layers to facilitate traversing)
        self.layers = layers
        self.classifier = nn.Sequential(*layers)

        if self.init_weights:
            self._initialize_weights(self.bias)

        # track correlations
        self.correlations = []

    def _kwinners(self, fout):
        return KWinners(
            n=fout,
            percent_on=self.percent_on,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
        )

    def _linear_block(self, fin, fout):
        block = [nn.Linear(fin, fout, bias=self.bias), self.activation_func(fout)]
        if self.batch_norm:
            block.append(nn.BatchNorm1d(fout))
        if self.dropout:
            block.append(nn.Dropout(p=self.dropout))

        return block

    def _has_activation(self, idx, layer):
        return (
            idx == len(self.layers) - 1
            or isinstance(layer, nn.ReLU)
            or isinstance(layer, KWinners)
        )

    def forward(self, x):
        """A faster and approximate way to track correlations"""
        x = x.view(-1, self.input_size)  # resiaze if needed, eg mnist
        prev_act = (x > 0).detach().float()
        idx_activation = 0
        for idx_layer, layer in enumerate(self.layers):
            # do the forward calculation normally
            x = layer(x)
            if self.hebbian_learning:
                n_samples = x.shape[0]
                if self._has_activation(idx_layer, layer):
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

    def _initialize_weights(self, bias):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if bias:
                    nn.init.constant_(m.bias, 0)


def vgg19_bn(config):
    model = models.vgg19_bn()
    # remove all fc layers, replace for a single fc layer, from 143mi to 20mi parameters
    model.classifier = nn.Linear(7 * 7 * 512, config["num_classes"])
    return model


def vgg19_bn_kw(config):
    model = models.vgg19_bn()
    # remove all fc layers, replace for a single fc layer, from 143mi to 20mi parameters
    model.classifier = nn.Linear(7 * 7 * 512, config["num_classes"])

    new_features = []
    for layer in model.features:
        # remove max pooling
        if isinstance(layer, nn.MaxPool2d):
            nn.AvgPool2d(kernel_size=2, stride=2)
        # store the number of out channels from conv layers
        elif isinstance(layer, nn.Conv2d):
            new_features.append(layer)
            last_conv_out_channels = layer.out_channels
        # switch ReLU to kWinners2d
        elif isinstance(layer, nn.ReLU):
            new_features.append(
                KWinners2d(
                    channels=last_conv_out_channels,
                    percent_on=config["percent_on"],
                    boost_strength=config["boost_strength"],
                    boost_strength_factor=config["boost_strength_factor"],
                )
            )
        # otherwise add it as normal
        else:
            new_features.append(layer)
    model.features = nn.Sequential(*new_features)

    return model


def resnet18(config):
    return models.resnet18(num_classes=config["num_classes"])


def resnet50(config):
    return models.resnet50(num_classes=config["num_classes"])


def vgg19_dsnn(config):

    net = VGG19(config=config)
    net = replace_sparse_weights(net)
    net = make_dsnn(net, config)
    net.classifier = swap_layers(
        net.classifier, nn.MaxPool2d, KWinners2d)
    net.classifier = squash_layers(
        net.classifier, DSConv2d, nn.BatchNorm2d, KWinners2d)
    net.classifier = squash_layers(
        net.classifier, DSConv2d, nn.BatchNorm2d, KWinners2d, nn.AvgPool2d)
    net.classifier = squash_layers(
        net.classifier, DSLinear, nn.BatchNorm1d, KWinners)

    if config.get("init_weights"):
        VGG19.initialize_weights(net.classifier)

    # Sanity check.
    # assert 

    return net
