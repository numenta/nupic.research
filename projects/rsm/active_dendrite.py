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

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from nupic.torch.modules.sparse_weights import SparseWeights


class LocalLinear(nn.Module):
    """
    """

    def __init__(self, in_features, local_features, kernel_size, stride=1, bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        fold_num = (in_features - self.kernel_size) // self.stride + 1
        self.lc = nn.ModuleList(
            [
                deepcopy(nn.Linear(kernel_size, local_features, bias=bias))
                for _ in range(fold_num)
            ]
        )

    def forward(self, x):
        x = x.unfold(-1, size=self.kernel_size, step=self.stride)
        fold_num = x.shape[1]
        x = torch.cat([self.lc[i](x[:, i, :]) for i in range(fold_num)], 1)
        return x


class ActiveDendriteLayer(torch.nn.Module):
    """
    Local layer for active dendrites. Similar to a non-shared weight version of a
    2D Conv layer.
    Note that dendrites are fully connected to input, local layer used
    only for connecting neurons and their dendrites

    Strategy:
        local_linear:
        maxpool:
    """

    def __init__(
        self, input_dim, n_cells=50, n_dendrites=3, strategy="maxpool", sparsity=0.3
    ):
        super(ActiveDendriteLayer, self).__init__()
        self.n_cells = n_cells
        self.n_dendrites = n_dendrites
        self.strategy = strategy

        total_dendrites = n_dendrites * n_cells
        self.linear_dend = SparseWeights(
            nn.Linear(input_dim, total_dendrites), sparsity
        )
        if self.strategy == "local_linear":
            self.linear_neuron = LocalLinear(
                total_dendrites, 1, n_dendrites, stride=n_dendrites
            )

    def __repr__(self):
        return "ActiveDendriteLayer neur=%d, dend per neuron=%d" % (
            self.n_cells,
            self.n_dendrites,
        )

    def forward(self, x):
        x = F.relu(self.linear_dend(x))
        if self.strategy == "local_linear":
            x = self.linear_neuron(x)
        elif self.strategy == "maxpool":
            x = x.view(-1, self.n_cells, self.n_dendrites).max(dim=2).values
        return x


if __name__ == "__main__":
    d_in = 10
    bsz = 2
    dends = ActiveDendriteLayer(d_in, n_cells=2, n_dendrites=3, strategy="maxpool")

    x = torch.rand(bsz, d_in)
    print("x", x)
    print("out", dends(x))
