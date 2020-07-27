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

"""
Dense/sparse MLP implementation
"""

import torch
import torch.nn.functional as F

from nupic.torch.modules import Flatten, KWinners, SparseWeights


class TwoLayerClassifier(torch.nn.Module):
    """ A dense/sparse MLP classifier with 2 hidden layers """

    def __init__(self, input_size=28 * 28, n_hidden_units=1000, n_classes=10,
                 is_sparse=False, sparsity=(0.75, 0.85), percent_on=0.1):
        """
        Initialize a 2-layer MLP
        :param input_size: number of input features to the MLP
        :type input_size: int
        :param n_hidden_units: number of units in each of the two hidden layers
        :type n_hidden_units: int
        :param n_classes: number of output units
        :type n_classes: int
        :param is_sparse: whether or not to initialize the sparse network instead of a
        dense one
        :type is_sparse: bool
        :param sparsity: a 2-element list/tuple specifying the sparsity in each of the
        hidden layers
        :type sparsity: list/tuple of float
        :param percent_on: number of active units in the K-Winners layer (only applies
        to sparse networks)
        :type percent_on: float
        """
        super().__init__()

        self.is_sparse = is_sparse
        self.flatten = Flatten()
        self.n_classes = n_classes

        self.fc1 = torch.nn.Linear(input_size, n_hidden_units)
        self.fc2 = torch.nn.Linear(n_hidden_units, n_hidden_units)
        self.fc3 = torch.nn.Linear(n_hidden_units, n_classes)

        if is_sparse:
            self.fc1_sparsity, self.fc2_sparsity = sparsity
            self.percent_on = percent_on

            self.fc1 = SparseWeights(self.fc1, sparsity=self.fc1_sparsity)
            self.kw1 = KWinners(n=n_hidden_units, percent_on=percent_on,
                                boost_strength=0.0)

            self.fc2 = SparseWeights(self.fc2, sparsity=self.fc2_sparsity)
            self.kw2 = KWinners(n=n_hidden_units, percent_on=percent_on,
                                boost_strength=0.0)

    def forward(self, x):
        output = self.fc1(self.flatten(x))
        output = self.kw1(output) if self.is_sparse else F.relu(output)
        output = self.fc2(output)
        output = self.kw2(output) if self.is_sparse else F.relu(output)
        output = self.fc3(output)
        return output
