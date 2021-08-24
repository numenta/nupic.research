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
A simple implementation of dendrite segments. This is meant to offer, at the least,
a template for revision and further modifications.
"""

import math
from collections.abc import Iterable
from itertools import product

import numpy as np
import torch
from torch.nn import init

from nupic.torch.modules.sparse_weights import HasRezeroWeights


class DendriteSegments(torch.nn.Module, HasRezeroWeights):
    """
    This implements dendrite segments over a set of units. Each unit has a set of
    segments modeled by a linear transformation from a context vector to output value
    for each segment.
    """

    def __init__(self, num_units, num_segments, dim_context, sparsity, bias=None):
        """
        :param num_units: number of units i.e. neurons;
                          each unit will have it's own set of dendrite segments
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param num_segments: number of dendrite segments per unit
        :param sparsity: sparsity of connections;
                        this is over each linear transformation from
                        dim_context to num_segments
        """
        super().__init__()

        # Save params.
        self.num_units = num_units
        self.num_segments = num_segments
        self.dim_context = dim_context
        self.sparsity = sparsity

        # TODO: Use named dimensions.
        weights = torch.Tensor(num_units, num_segments, dim_context)
        self.weights = torch.nn.Parameter(weights)

        # Create a bias per unit per segment.
        if bias:
            biases = torch.Tensor(num_units, num_segments)
            self.biases = torch.nn.Parameter(biases)
        else:
            self.register_parameter("biases", None)
        self.reset_parameters()

        # Create a random mask per unit per segment (dims=[0, 1])
        zero_mask = random_mask(
            self.weights.shape,
            sparsity=sparsity,
            dims=[0, 1]
        )

        # Use float16 because pytorch distributed nccl doesn't support bools.
        self.register_buffer("zero_mask", zero_mask.half())

        self.rezero_weights()

    def extra_repr(self):
        return (
            f"num_units={self.num_units}, "
            f"num_segments={self.num_segments}, "
            f"dim_context={self.dim_context}, "
            f"sparsity={self.sparsity}, "
            f"bias={self.biases is not None}"
        )

    def reset_parameters(self):
        """Initialize the linear transformation for each unit."""
        for unit in range(self.num_units):
            weight = self.weights[unit, ...]
            if self.biases is not None:
                bias = self.biases[unit, ...]
            else:
                bias = None
            init_linear_(weight, bias)

    def rezero_weights(self):
        self.weights.data.masked_fill_(self.zero_mask.bool(), 0)

    def forward(self, context):
        """
        Matrix-multiply the context with the weight tensor for each dendrite segment.
        This is done for each unit and so the output is of length num_units.
        """

        # Matrix multiply using einsum:
        #    * b => the batch dimension
        #    * k => the context dimension; multiplication will be along this dimension
        #    * ij => the units and segment dimensions, respectively
        # W^C * M^C * C -> num_units x num_segments
        output = torch.einsum("ijk,bk->bij", self.weights, context)

        if self.biases is not None:
            output += self.biases
        return output


def init_linear_(weight, bias=None):
    """
    Performs the default initilization of a weight and bias parameter
    of a linear layaer; done in-place.
    """
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


def random_mask(size, sparsity, dims=None, **kwargs):
    """
    This creates a random off-mask (True => off) of 'size' with the specified 'sparsity'
    level along 'dims'. If 'dims' is 1, for instance, then `mask[:, d, ...]` has the
    desired sparsity for all d. If dims is a list, say [0, 1], then `mask[d1, d2, ...]`
    will have the desired sparsity level for all d1 and d2. If None, the sparsity is
    applied over the whole tensor.

    :param size: shape of tensor
    :param sparsity: fraction of non-zeros
    :param dims: which dimensions to apply the sparsity
    :type dims: int or iterable
    :param kwargs: keywords args passed to torch.ones;
                   helpful for specifying device, for instace
    """

    assert 0 <= sparsity <= 1

    # Start with all elements off.
    mask = torch.ones(size, **kwargs)

    # Find sparse submasks along dims; recursively call 'random_mask'.
    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = [dims]

        # Loop all combinations that index through dims.
        # The 1D case is equivalent to range.
        dim_lengths = [mask.shape[dim] for dim in dims]
        dim_indices = product(*[range(dl) for dl in dim_lengths])

        for idxs in dim_indices:

            # For example, this may yield a slice that gives
            # `mask[dim_slice] == mask[:, 0, 0]` where `dims=[1, 2]`.
            dim_slice = [
                idxs[dims.index(d)] if d in dims else slice(None)
                for d in range(len(mask.shape))
            ]

            # Assign the desired sparsity to the submask.
            sub_mask = mask[dim_slice]
            sub_mask[:] = random_mask(
                sub_mask.shape,
                sparsity, **kwargs, dims=None
            )

        return mask

    # Randomly choose indices to make non-zero ("nz").
    mask_flat = mask.view(-1)  # flattened view
    num_total = mask_flat.shape[0]
    num_nz = int(round((1 - sparsity) * num_total))
    on_indices = np.random.choice(num_total, num_nz, replace=False)
    mask_flat[on_indices] = False

    return mask
