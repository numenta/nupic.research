# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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


class DQNNetwork(nn.Module):
    def __init__(self, input_channels=4, num_actions=18):
        """
        Initialize a deep Q-learning network
        Arguments:
            input_channels: number of channel of input.
                            In DQN, can correspond to number of stacked frames
            num_actions: number of action-value to output, one-to-one correspondence to
                         action in game.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
        )
        self.classifier = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_action(self, x):
        return self.forward(x).argmax().item()
