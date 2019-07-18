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

import math

import torch
from torch import nn
from torchvision import models

from nupic.torch.modules import Flatten, KWinners2d


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


class VGG19(nn.Module):
    def __init__(self, config=None):
        super(VGG19, self).__init__()

        defaults = dict(
            input_size=784,
            num_classes=10,
            hidden_sizes=[4000, 1000, 4000],
            batch_norm=False,
            dropout=0.3,
            bias=False,
            init_weights=True,
            kwinners=False,
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
        layers.append(nn.Linear(512, config["num_classes"]))
        self.classifier = nn.Sequential(*layers)

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

    def forward(self, x):
        return self.classifier(x)

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


class MLP(nn.Module):
    """
    Simple 3 hidden layers + output MLP, similar to one used in SET Paper.
    """

    def __init__(self, config=None):
        super(MLP, self).__init__()

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
