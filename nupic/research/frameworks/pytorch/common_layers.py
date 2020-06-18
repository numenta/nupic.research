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

from nupic.research.frameworks.pytorch.modules import (
    MaskedConv2d,
    MaskedLinear,
    maskedconv2d_init,
    maskedlinear_init,
)
from nupic.torch.modules import KWinners2d, SparseWeights, SparseWeights2d


def always_dense(*args, **kwargs):
    return 1.0


def relu_maybe_kwinners2d(channels,
                          density_fn=always_dense,
                          k_inference_factor=1.0,
                          boost_strength=1.0,
                          boost_strength_factor=0.9,
                          duty_cycle_period=1000,
                          local=True):
    """
    Get a nn.ReLU, possible followed by a KWinners2d
    """
    layer = nn.ReLU(inplace=True)
    density = density_fn(channels)
    if density < 1.0:
        layer = nn.Sequential(
            layer,
            KWinners2d(channels, percent_on=density,
                       boost_strength=boost_strength,
                       boost_strength_factor=boost_strength_factor,
                       local=local, k_inference_factor=k_inference_factor)
        )
    return layer


def sparse_linear(in_features, out_features, bias=True,
                  density_fn=always_dense):
    """
    Get a nn.Linear, possibly wrapped in a SparseWeights
    """
    layer = nn.Linear(in_features, out_features, bias=bias)
    density = density_fn(in_features, out_features)
    if density < 1.0:
        layer = SparseWeights(layer, weight_sparsity=density)
    return layer


def sparse_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
                  dilation=1, groups=1, bias=True, density_fn=always_dense):
    """
    Get a nn.Conv2d, possibly wrapped in a SparseWeights2d
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups,
                      bias=bias)
    density = density_fn(in_channels, out_channels, kernel_size)
    if density < 1.0:
        layer = SparseWeights2d(layer, weight_sparsity=density)
    return layer


def masked_linear(in_features, out_features, bias=True, density_fn=always_dense):
    """
    Get a MaskedLinear, initialized with a particular density
    """
    density = density_fn(in_features, out_features)
    if density < 1.0:
        layer = MaskedLinear(in_features, out_features, bias=bias)
        maskedlinear_init(layer, density)
    else:
        layer = nn.Linear(in_features, out_features, bias=bias)
    return layer


def masked_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
                  dilation=1, groups=1, bias=True, density_fn=always_dense):
    """
    Get a MaskedConv2d, initialized with a particular density
    """
    density = density_fn(in_channels, out_channels, kernel_size)
    if density < 1.0:
        layer = MaskedConv2d(in_channels, out_channels, kernel_size, stride=stride,
                             padding=padding, dilation=dilation, groups=groups,
                             bias=bias)
        maskedconv2d_init(layer, density)
    else:
        layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=groups,
                          bias=bias)
    return layer
