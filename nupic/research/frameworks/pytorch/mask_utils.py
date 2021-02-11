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
