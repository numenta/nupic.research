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
Convenience functions that construct modules or combinations of modules.

These functions are designed to be called in places where networks are being
defined by data.
"""

from torch import nn

from nupic.torch.modules import KWinners2d, SparseWeights, SparseWeights2d


def relu_maybe_kwinners2d(channels,
                          density=1.0,
                          k_inference_factor=1.0,
                          boost_strength=1.0,
                          boost_strength_factor=0.9,
                          duty_cycle_period=1000,
                          local=True,
                          inplace=True,
                          break_ties=False):
    """
    Get a nn.ReLU, possible followed by a KWinners2d

    :param density:
        Either a density or a function that returns a density.
    :type density: float or function(channels)
    """
    if callable(density):
        density = density(channels)

    if density == 1.0:
        return nn.ReLU(inplace=inplace)

    if break_ties:
        return nn.Sequential(
            nn.ReLU(inplace=inplace),
            KWinners2d(channels, percent_on=density,
                       k_inference_factor=k_inference_factor,
                       boost_strength=boost_strength,
                       boost_strength_factor=boost_strength_factor,
                       local=local, break_ties=True))
    else:
        # Use the KWinners2d built-in ReLU functionality. Use a module structure
        # that is checkpoint-compatible with the above case for
        # cross-compatibility and backward-compatibility.
        return nn.Sequential(
            nn.Identity(),
            KWinners2d(channels, percent_on=density,
                       k_inference_factor=k_inference_factor,
                       boost_strength=boost_strength,
                       boost_strength_factor=boost_strength_factor,
                       local=local, break_ties=False, inplace=inplace,
                       relu=True))


def sparse_linear(in_features, out_features, bias=True, density=1.0):
    """
    Get a nn.Linear, possibly wrapped in a SparseWeights

    :param density:
        Either a density or a function that returns a density.
    :type density: float or function(in_features, out_features)
    """
    layer = nn.Linear(in_features, out_features, bias=bias)

    if callable(density):
        density = density(in_features, out_features)

    if density < 1.0:
        layer = SparseWeights(layer, weight_sparsity=density)
    return layer


def sparse_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
                  dilation=1, groups=1, bias=True, density=1.0):
    """
    Get a nn.Conv2d, possibly wrapped in a SparseWeights2d

    :param density:
        Either a density or a function that returns a density.
    :type density: float or function(in_channels, out_channels, kernel_size)
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups,
                      bias=bias)

    if callable(density):
        density = density(in_channels, out_channels, kernel_size)

    if density < 1.0:
        layer = SparseWeights2d(layer, weight_sparsity=density)
    return layer


__all__ = [
    "relu_maybe_kwinners2d",
    "sparse_linear",
    "sparse_conv2d",
]
