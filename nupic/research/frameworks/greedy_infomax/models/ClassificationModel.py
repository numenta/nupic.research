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


class ClassificationModel(nn.Module):
    def __init__(self, in_channels=256, num_classes=10, hidden_nodes=0):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AvgPool2d((7, 7), padding=0)
        self.model = nn.Sequential()
        self.model.add_module(
            "layer1", nn.Linear(self.in_channels, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.avg_pool(x).squeeze()
        x = self.model(x).squeeze()
        return x
