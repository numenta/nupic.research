# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
#
# This work was built off the Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The original Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------

import torch.nn as nn


class Classifier(nn.Module):
    """
    A simple multilayer perceptron classification head which outputs a distribution
    over possible class labels. This is used in the supervised phase of Greedy
    InfoMax experiments to tell how "useful" a given encoding is by proxy of how
    well an encoding can be mapped to the correct class label.

    :param in_channels: The dimensionality of the input to this model
    :param num_classes: The dimensionality of the output (the number of possible
    class labels)

    """
    def __init__(self, in_channels=256, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AvgPool2d((7, 7), padding=0)
        self.model = nn.Sequential()
        self.model.add_module(
            "layer1", nn.Linear(self.in_channels, num_classes, bias=True)
        )

    def forward(self, x):
        # detach x just in case it's still connected to active parts of the
        # computation graph
        batch_size = x.shape[0]
        x = x.detach()
        x = self.avg_pool(x).squeeze()
        x = self.model(x).squeeze()
        return x.view(batch_size, -1)


class MultiClassifier(nn.Module):
    """
    A model which contains many classification models. Oftentimes, a Greedy InfoMax
    experiment that uses the BlockModel will emit several different encodings,
    one for each EmitEncoding layer. This model allows for a classifier to be
    independently fit to each encoding layer, which shows how well each layer can be
    used on a downstream task.

    """
    def __init__(self, in_channels=None, num_classes=10):
        super().__init__()
        if self.in_channels is None:
            raise Exception("In channels list is required")
        self.classifiers = nn.ModuleList(
            [
                Classifier(
                    in_channels=in_channels[i],
                    num_classes=num_classes,
                )
                for i in range(len(in_channels))
            ]
        )

    def forward(self, encodings):
        return [
            classifier(encoding)
            for (classifier, encoding) in zip(self.classifiers, encodings)
        ]
