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

import torch


def indices_to_mask(indices, shape, dim):
    """
    Constructs a mask of ones corresponding the given indices along the desired
    dimension.

    For a 3-D tensor the output is specified by:
    ```
    mask[indices[i][j][k]][j][k] = 1  # if dim == 0
    mask[i][indices[i][j][k]][k] = 1  # if dim == 1
    mask[i][j][indices[i][j][k]] = 1  # if dim == 2
    ```

    :param indices: torch Tensor of indices, such as those returned from torch.max
    :param shape: the shape of the final mask
    :param dim: which dimension the indices refer to
    """

    # Validate the desired shape is the same as `indices` but with the dimension `dim`
    # squeezed out.
    assert indices.shape == torch.Size([s for i, s in enumerate(shape) if i != dim])

    # The 'mask' is zeros by default and 'ones' is a helper tensor for tallying.
    mask = torch.zeros(shape,
                       dtype=bool,
                       layout=indices.layout,
                       device=indices.device)
    ones = torch.ones_like(mask)

    # Every location within 'indices' will give a one to the 'mask'.
    mask.scatter_(dim=dim, index=indices.unsqueeze(dim), src=ones)
    return mask


def get_topk_submask(k, values, mask, largest=True):
    """
    The function finds the top values within the `mask` and returns a new sub-mask
    indicating the positions of these top values.

    :param k:
        int indicating the number of values to select
    :param values:
        torch.Tensor of sortable values
    :param mask:
        torch.BoolTensor indicating which subset of `values` to chose from
    :param largest:
        bool - whether to select the largest or smallest values
               this parameter is passed to torch.topk and thus has a similar meaning

    :return:
        torch.BoolTensor mask indicating the positions of the `percentage` values
    """

    assert values.shape == mask.shape

    # Focus on the magnitude of the values which are on.
    on_indices = mask.nonzero(as_tuple=True)
    on_values = values[on_indices]

    # Find the top values which are on.
    _, top_on_sub_indices = on_values.topk(k, largest=largest)

    # Convert those indices into those for the larger `values` matrix.
    top_all_indices = [
        on_idxs[top_on_sub_indices].cpu().numpy() for on_idxs in on_indices
    ]

    # Construct a mask of the top positive values.
    top_values_mask = torch.zeros_like(values, dtype=torch.bool)
    top_values_mask[top_all_indices] = 1

    return top_values_mask
