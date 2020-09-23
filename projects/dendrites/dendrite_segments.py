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

import numpy as np
import torch
from torch.nn import init


class DendriteSegments(torch.nn.Module):
    """
    This implements dendrite segments over a set of units. Each unit has a set of
    segments modeled by a linear transformation from a context vector to output value
    for each segment.

    :param num_units: number of units i.e. neurons;
                      each unit will have it's own set of dendrite segments
    :param num_context: length of the context vector;
                        the same context will be applied to each segment
    :param num_segments: number of dendrite segments per unit
    :param sparsity: sparsity of connections;
                     this is over each linear transformation from
                     num_context to num_segments

    TODO: Should we include a bias?
    """

    def __init__(self, num_units, num_segments, num_context, sparsity):
        super().__init__()

        segment_weights = torch.zeros(num_units, num_segments, num_context)
        self.segment_weights = torch.nn.Parameter(segment_weights)

        # TODO: What initialization should we use?
        init.kaiming_uniform_(self.segment_weights, a=math.sqrt(5))

        # Create a random mask for each unit (dim=0)
        zero_mask = random_mask(self.segment_weights.shape, sparsity=sparsity, dim=0)

        # Use float16 because pytorch distributed nccl doesn't support bools.
        self.register_buffer("zero_mask", zero_mask.half())

        # Save params.
        self.num_units = num_units
        self.num_segments = num_segments
        self.num_context = num_context
        self.sparsity = sparsity

    def rezero_weights(self):
        self.segment_weights.data[self.zero_mask.bool()] = 0

    def forward(self, context):
        """
        Matrix-multiply the context with the weight tensor for each dendrite segment.
        This is done for each unit and so the output is of length num_units.
        """
        return apply_context(context, self.segment_weights)


def apply_context(context, segment_weights):
    """
    Applies context vector to a set of weighted dendrite segments. This performs
    as tensor multiplication of segment_weights x context.

    :param context: context tensor of shape minibatch x num_context
    :param segment_weights: dendrite segment weights of shape
                            num_units x num_segments x num_context
    """

    assert len(context.shape) == 2, "Expected context of shape minibatch x num_context"
    assert len(segment_weights.shape) == 3, (
        "Expected segment-weights of shape num_units x num_segments x num_context."
    )
    assert context.shape[1] == segment_weights.shape[2], (
        "The outter dims of 'context' and 'segment_weights should be equal.'"
    )

    # Matrix multiply using einsum:
    #    * b => the batch dimension
    #    * k => the context dimension; multiplication will be along this dimension
    #    * ij => the units and segment dimensions, respectively
    out = torch.einsum("ijk,bk->bij", segment_weights, context)

    # TODO: Test if the following is faster
    #       out = torch.tensordot(param, context.transpose(1, 0), dims=1)
    #       out = out.permute(2, 0, 1)
    return out


def random_mask(size, sparsity, dim=None, **kwargs):
    """
    This creates a random off-mask (True => off) of 'size' with the
    specified 'sparsity' level along 'dim'. If 'dim' is 1, for instance,
    then `mask[:, d, ...]` has the desired sparsity for all d. If None,
    the sparsity is applied over the whole tensor.

    :param size: shape of tensor
    :param sparsity: fraction of non-zeros
    :param dim: which dimension to apply the sparsity
    :param kwargs: keywords args passed to torch.ones;
                   helpful for specifying device, for instace
    """

    assert 0 <= sparsity <= 1

    # Start with all elements off.
    mask = torch.ones(size, **kwargs)

    # Find sparse submasks along dim; recursively call 'random_mask'.
    if dim is not None:
        len_of_dim = mask.shape[dim]
        for d in range(len_of_dim):
            dim_slice = [slice(None)] * len(mask.shape)
            dim_slice[dim] = d
            sub_mask = mask[dim_slice]
            sub_mask[:] = random_mask(
                sub_mask.shape,
                sparsity, **kwargs, dim=None
            )
        return mask

    # Randomly choose indices to make non-zero ("nz").
    mask_flat = mask.view(-1)  # flattened view
    num_total = mask_flat.shape[0]
    num_nz = int(round((1 - sparsity) * num_total))
    on_indices = np.random.choice(num_total, num_nz, replace=False)
    mask_flat[on_indices] = False

    return mask


if __name__ == "__main__":

    dendrite_segment = DendriteSegments(
        num_units=10, num_segments=20, num_context=15, sparsity=0.7
    )
    dendrite_segment.rezero_weights()

    batch_size = 8
    context = torch.rand(batch_size, dendrite_segment.num_context)
    out = dendrite_segment(context)

    print(f"out.shape={out.shape}")
