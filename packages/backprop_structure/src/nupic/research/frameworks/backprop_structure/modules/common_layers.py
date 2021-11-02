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

from nupic.research.frameworks.backprop_structure.modules import (
    VDropConv2d,
    VDropLinear,
)


def prunable_vdrop_linear(in_features, out_features, vdrop_data, bias=True,
                          target_density=1.0):
    layer = VDropLinear(in_features, out_features, vdrop_data, bias)

    if callable(target_density):
        target_density = target_density(in_features, out_features)

    if target_density < 1.0:
        layer._target_density = target_density

    return layer


def prunable_vdrop_conv2d(in_channels, out_channels, kernel_size, vdrop_data,
                          stride=1, padding=0, dilation=1, groups=1, bias=True,
                          target_density=1.0):
    layer = VDropConv2d(in_channels, out_channels, kernel_size, vdrop_data,
                        stride, padding, dilation, groups, bias)

    if callable(target_density):
        target_density = target_density(in_channels, out_channels, kernel_size)

    if target_density < 1.0:
        layer._target_density = target_density

    return layer


__all__ = [
    "prunable_vdrop_linear",
    "prunable_vdrop_conv2d",
]
