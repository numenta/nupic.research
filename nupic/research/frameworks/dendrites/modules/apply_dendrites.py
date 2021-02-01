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

"""
Contains functions for applying dendrite weights. These functions each implement
a method to combine the output from a (sparse) linear layer with the output from
a set of dendritic segments.
"""

import torch

from nupic.research.frameworks.dendrites import functional as F

__all__ = [
    "DendriticBias1d",
    "DendriticGate1d",
    "DendriticAbsoluteMaxGate1d",
    "DendriticGate2d",
    "DendriticAbsoluteMaxGate2d",
]


class DendriticBias1d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return F.dendritic_bias_1d(y, dendrite_activations)


class DendriticGate1d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return F.dendritic_gate_1d(y, dendrite_activations)


class DendriticAbsoluteMaxGate1d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return F.dendritic_absolute_max_gate_1d(y, dendrite_activations)


class DendriticGate2d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return F.dendritic_gate_2d(y, dendrite_activations)


class DendriticAbsoluteMaxGate2d(torch.nn.Module):
    def forward(self, y, dendrite_activations):
        return F.dendritic_absolute_max_gate_2d(y, dendrite_activations)
