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

    def __init__(self, input_size, num_classes):

        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(input_size)), 100),
            nn.Linear(100, 100),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
