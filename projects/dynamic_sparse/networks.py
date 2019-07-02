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
from torchvision import models


def vgg19_bn(config):
    model = models.vgg19_bn()
    # remove all fc layers, replace for a single fc layer, from 143mi to 20mi parameters
    model.classifier = nn.Linear(7 * 7 * 512, config["num_classes"])
    return model


def resnet18(config):
    return models.resnet18(num_classes=config["num_classes"])


def resnet50(config):
    return models.resnet50(num_classes=config["num_classes"])


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
        if config is None:
            config = {}
        defaults.update(config)
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
