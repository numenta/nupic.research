# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from collections import namedtuple

# Defines sparse params for regular layers with activations
# If params_function is not None, it should be called to calculate the actual
# parameters. Useful for large networks such as ResNet where it's hard to tune
# each layer manually.
LayerParams = namedtuple(
    "LayerParams",
    [
        "percent_on",
        "boost_strength",
        "boost_strength_factor",
        "k_inference_factor",
        "local",
        "weights_density",
        "params_function",
    ],
)

# Defaults to a dense layer
LayerParams.__new__.__defaults__ = (1.0, 1.4, 0.7, 1.0, True, 1.0, None)

# Defines default sparse params for layers without activations
NoactLayerParams = namedtuple("NoactLayerParams",
                              ["weights_density", "params_function"])
NoactLayerParams.__new__.__defaults__ = (1.0, None)


def auto_sparse_params(in_channels, out_channels, kernel_size):
    """
    Given conv2d parameters, automatically calculate sparsity parameters.
    This is highly experimental and likely to change.

    :return: an instance of LayerParams
    """
    weights_per_channel = kernel_size * kernel_size * in_channels
    if weights_per_channel < 100:
        weights_density = 0.75

    elif weights_per_channel < 200:
        weights_density = 0.5

    elif weights_per_channel < 500:
        weights_density = 0.4

    elif weights_per_channel < 1000:
        weights_density = 0.3

    elif weights_per_channel < 2000:
        weights_density = 0.2

    elif weights_per_channel < 4000:
        weights_density = 0.2

    else:
        weights_density = 0.2

    if kernel_size != 1:
        if out_channels > 128:
            percent_on = 0.3
        else:
            percent_on = 1.0
    else:
        percent_on = 1.0

    return LayerParams(
        percent_on=percent_on,
        boost_strength=1.0,
        boost_strength_factor=0.9,
        local=True,
        weights_density=weights_density,
        params_function=None
    )
