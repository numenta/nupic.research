# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import torch.nn as nn

from nupic.research.frameworks.pytorch.model_utils import filter_params


class ANMLNetwork(nn.Module):
    """
    This network is reproduced from the `ANML`_ repository. This expects
    an input size of 28 x 28 (which differs from OML's and ours wherein
    we reshape Omniglot images to 84 x 84).

    .. _ANML: https://github.com/uvm-neurobotics-lab/ANML

    With `num_classes=963`, it uses 5,963,139 weights in total.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.prediction = nn.Sequential(
            *self.conv_block(3, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.Flatten(),
        )
        self.neuromodulation = nn.Sequential(
            *self.conv_block(3, 112, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(112, 112, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(112, 112, 3, 1, 0),
            nn.Flatten(),
            nn.Linear(1008, 2304),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(2304, num_classes)
        self.reset_params()

        # The ANML repo always keep its model in training mode. See self.eval()
        self.train()

    def reset_params(self):
        # Apply Kaiming initialization to Linear and Conv2d params
        named_params = filter_params(self, include_modules=[nn.Linear, nn.Conv2d])
        for _, param in named_params.items():
            if param.ndim > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    @classmethod
    def conv_block(cls, in_channels, out_channels, kernel_size, stride, padding):
        return [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(),
        ]

    def forward(self, x):
        pred = self.prediction(x)
        mod = self.neuromodulation(x)
        out = pred * mod
        out = self.classifier(out)
        return out


class ANMLsOMLNetwork(nn.Module):
    """
    ANML's implementation of OML's network. This differs from the OML repo
    in that it has two linear layers in the prediction network as opposed to one.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.representation = nn.Sequential(
            *self.conv_block(3, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 2, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 1, 0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *self.conv_block(256, 256, 3, 2, 0),
            nn.Flatten(),
        )
        self.prediction = nn.Sequential(
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

        self.reset_params()

    def reset_params(self):
        # Apply Kaiming initialization
        for param in self.parameters():
            if param.ndim > 1:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)

    @classmethod
    def conv_block(cls, in_channels, out_channels, kernel_size, stride, padding):
        return [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.ReLU(),
        ]

    def forward(self, x):
        out = self.representation(x)
        out = self.prediction(x)
        return out
