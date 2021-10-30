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

from collections import namedtuple
from typing import Optional

import torch

__all__ = [
    "dendrite_output",
    "dendritic_bias_1d",
    "dendritic_gate_1d",
    "dendritic_absolute_max_gate_1d",
    "dendritic_gate_2d",
    "dendritic_absolute_max_gate_2d",
]


dendrite_output = namedtuple("dendrite_output", ["values", "indices"])
dendrite_output.__doc__ = """
A named tuple for outputs modified by `apply_dendrites`_.

:attr values: output tensor after being modulated by dendrite activations
:attr indices: the indices of the winning segments used to modulate the output tensor

.. _apply_dendrites: nupic.research.frameworks.dendrites.functional.apply_dendrites
"""


@torch.jit.script
def dendritic_bias_1d(y, dendrite_activations):
    """
    Returns the sum of the feedforward output and the max of the dendrite
    activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments respectively.
    """
    # Take max along each segment.
    winning_activations, indices = dendrite_activations.max(dim=2)
    return dendrite_output(y + winning_activations, indices)


@torch.jit.script
def gather_activations(dendrite_activations, indices):
    """
    Gathers dendritic activations from the given indices.

    :param indices: tensor of indices of winning segments;
                    shape of batch_size x num_units
    :param indices: tensor of dendritic activations;
                    shape of batch_size x num_units x num_segments
    """
    unsqueezed = indices.unsqueeze(dim=2)
    dendrite_activations = torch.gather(dendrite_activations, dim=2, index=unsqueezed)
    dendrite_activations = dendrite_activations.squeeze(dim=2)
    return dendrite_activations


@torch.jit.script
def dendritic_gate_1d(y, dendrite_activations, indices: Optional[torch.Tensor] = None):
    """
    Returns the product of the feedforward output and sigmoid of the the max
    of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments, respectively.
    :param indices: (optional) indices of winning segments;
                    shape of batch_size x num_units
    """
    # Select winner by max activations, or use given indices as winners.
    if indices is None:
        winning_activations, indices = dendrite_activations.max(dim=2)
    else:
        winning_activations = gather_activations(dendrite_activations, indices)

    # Multiple by the sigmoid of the max along each segment.
    return dendrite_output(y * torch.sigmoid(winning_activations), indices)


@torch.jit.script
def dendritic_absolute_max_gate_1d(y, dendrite_activations):
    """
    Returns the product of the feedforward output and the sigmoid of the
    absolute max of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              size and number of units, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, s) where the
                                 axes represent the batch size, number of units, and
                                 number of segments, respectively.
    """
    indices = dendrite_activations.abs().max(dim=2).indices
    return dendritic_gate_1d(y, dendrite_activations, indices=indices)


@torch.jit.script
def dendritic_gate_2d(y, dendrite_activations, indices: Optional[torch.Tensor] = None):
    """
    Returns the output of the max gating convolutional dendritic layer by
    multiplying all values in each output channel by the selected dendrite
    activations. Dendrite activations are selected based on the maximum
    activations (keeping the sign) across all segments for each channel. Each
    channel has its own set of dendritic weights, and the selected activation is
    based on the the max value.

    :param y: output of the convolution operation (a torch tensor with shape
              (b, c, h, w) where the axes represent the batch, channel, height, and
              width dimensions respectively)
    :param dendrite_activations: the dendrite activation values (a torch tensor
                                 with shape (b, c, d) where the axes represent the
                                 batch size, number of channels, and number of segments
                                 respectively)
    :param indices: (optional) indices of winning segments;
                    shape of batch_size x num_units
    """
    if indices is None:
        winning_activations, indices = dendrite_activations.max(dim=2)
    else:
        winning_activations = gather_activations(dendrite_activations, indices)

    # The following operation uses `torch.einsum` to multiply each channel by a
    # single scalar value
    #    * b => the batch dimension
    #    * i => the channel dimension
    #    * jk => the width and height dimensions

    sigmoid_activations = torch.sigmoid(winning_activations)
    y_gated = torch.einsum("bijk,bi->bijk", y, sigmoid_activations)
    return dendrite_output(y_gated, indices)


@torch.jit.script
def dendritic_absolute_max_gate_2d(y, dendrite_activations):
    """
    Returns the output of the absolute max gating convolutional dendritic layer by
    multiplying all values in each output channel by the selected dendrite
    activations. Dendrite activations are selected based on the absolute maximum
    activations (keeping the sign) across all segments for each channel. Each
    channel has its own set of dendritic weights, and the selected activation is
    based on the the absolute max value.

    :param y: output of the convolution operation (a torch tensor with shape
              (b, c, h, w) where the axes represent the batch, channel, height, and
              width dimensions respectively)
    :param dendrite_activations: the dendrite activation values (a torch tensor
                                 with shape (b, c, d) where the axes represent the
                                 batch size, number of channels, and number of segments
                                 respectively)
    """
    indices = dendrite_activations.abs().max(dim=2).indices
    return dendritic_gate_2d(y, dendrite_activations, indices=indices)
