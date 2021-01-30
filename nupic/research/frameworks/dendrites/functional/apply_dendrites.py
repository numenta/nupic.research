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
Contains functions for applying dendrite weights. These functions each implement
a method to combine the output from a (sparse) linear layer with the output from
a set of dendritic segments.
"""

import torch


def dendritic_bias_1d(y, dendrite_activations):
    """
    Returns the sum of the feedforward output and the max of the dendrite
    activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the batch
              and vector dimensions, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, d) where the
                                 axes represent batch, vector, and dendrite
                                 dimensions, respectively.
    """
    return y + dendrite_activations.max(dim=2).values  # max along each segment


def dendritic_gate_1d(y, dendrite_activations):
    """
    Returns the product of the feedforward output and sigmoid of the the max
    of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent the
              batch and vector dimensions, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, d) where the
                                 axes represent batch, vector, and dendrite
                                 dimensions, respectively.
    """
    # Multiple by the sigmoid of the max along each segment.
    return y * torch.sigmoid(dendrite_activations.max(dim=2).values)


def dendritic_absolute_max_gate_1d(y, dendrite_activations):
    """
    Returns the product of the feedforward output and the sigmoid of the
    absolute max of the dendrite activations along each segment.
    :param y: torch Tensor with shape (b, n) where the axes represent
              the batch and vector dimensions, respectively.
    :param dendrite_activations: torch Tensor with shape (b, n, d) where
                                 the axes represent batch, vector, and
                                 dendrite dimensions, respectively.
    """
    inds = dendrite_activations.abs().max(dim=2).indices
    inds = inds.unsqueeze(dim=2)
    dendrite_activations = torch.gather(dendrite_activations, dim=2, index=inds)
    dendrite_activations = dendrite_activations.squeeze()
    dendrite_activations = torch.sigmoid(dendrite_activations)
    return y * dendrite_activations


def dendritic_gate_2d(y, dendrite_activations):
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
                                 with shape (b, c) where the axes represent the
                                 batch and channel dimensions, respectively)
    """
    dendrite_activations = dendrite_activations.max(dim=2).values
    dendrite_activations = torch.sigmoid(dendrite_activations)

    # The following operation uses `torch.einsum` to multiply each channel by a
    # single scalar value
    #    * b => the batch dimension
    #    * i => the channel dimension
    #    * jk => the width and height dimensions

    return torch.einsum("bijk,bi->bijk", y, dendrite_activations)


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
                                 with shape (b, c) where the axes represent the
                                 batch and channel dimensions, respectively)
    """
    inds = dendrite_activations.abs().max(dim=2).indices
    inds = inds.unsqueeze(dim=2)
    dendrite_activations = torch.gather(dendrite_activations, dim=2, index=inds)
    dendrite_activations = dendrite_activations.squeeze(dim=2)
    dendrite_activations = torch.sigmoid(dendrite_activations)

    # The following operation uses `torch.einsum` to multiply each channel by a
    # single scalar value
    #    * b => the batch dimension
    #    * i => the channel dimension
    #    * jk => the width and height dimensions

    return torch.einsum("bijk,bi->bijk", y, dendrite_activations)


__all__ = [
    "dendritic_bias_1d",
    "dendritic_gate_1d",
    "dendritic_absolute_max_gate_1d",
    "dendritic_gate_2d",
    "dendritic_absolute_max_gate_2d",
]
