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
# ----------------------------------------------------------------------
import torch.nn as nn

from nupic.research.frameworks.greedy_infomax.models.resnet_encoder import (
    PreActBlockNoBN,
)


class ResNet7(nn.Sequential):
    def __init__(self,
                 channels=64,):
        super(ResNet7, self).__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5, stride=1, padding=2)
        self.sparse_preact_1 = PreActBlockNoBN(channels, channels)
        self.sparse_preact_2 = PreActBlockNoBN(channels, channels)
        self.sparse_preact_3 = PreActBlockNoBN(channels, channels)
