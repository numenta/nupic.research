#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import numpy as np
import torch.nn as nn


class StandardMLP(nn.Module):

    def __init__(self, input_size, num_classes,
                 hidden_sizes=(100, 100)):

        super().__init__()

        layers = [
            nn.Flatten(),
            nn.Linear(int(np.prod(input_size)), hidden_sizes[0]),
            nn.ReLU()
        ]
        for idx in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(hidden_sizes[idx - 1], hidden_sizes[idx]),
                nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class SmallCNN(nn.Module):

    def __init__(self, num_classes, **kwargs):

        super().__init__()

        self.features = nn.Sequential(
            *self.conv_block(1, 8, 5),  # 105 -> 53
            *self.conv_block(8, 16, 3),  # 53 - 27
            *self.conv_block(16, 32, 3),  # 27 - 14
            *self.conv_block(32, 64, 3),  # 14 - 7
            *self.conv_block(64, 128, 3),  # 7 - 4
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 100),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    @classmethod
    def conv_block(cls, in_channels, out_channels, kernel_size):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        ]


class OMLNetwork(nn.Module):

    def __init__(self, num_classes, **kwargs):

        super().__init__()

        self.representation = nn.Sequential(
            *self.conv_block(1, 8, 5),  # 105 -> 53
            *self.conv_block(8, 16, 3),  # 53 - 27
            *self.conv_block(16, 32, 3),  # 27 - 14
            *self.conv_block(32, 64, 3),  # 14 - 7
            *self.conv_block(64, 128, 3),  # 7 - 4
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.adaptation = nn.Sequential(
            nn.Linear(128, 100),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.representation(x)
        x = self.adaptation(x)
        return x

    @classmethod
    def conv_block(cls, in_channels, out_channels, kernel_size):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        ]
