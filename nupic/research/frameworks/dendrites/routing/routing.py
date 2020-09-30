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

import torch

from nupic.torch.modules import SparseWeights
from utils import generate_random_binary_vectors


class RoutingFunction(torch.nn.Module):
    """
    A class to represent the routing function R(j, x) which computes a sparse linear
    transformation of x, followed by an element-wise mask specified by j, which indexes
    one of k binary masks particular to R.

    More formally, R(j, x) = T_j * Wx, where Wx is a sparse linear transformation of
    the input to the routing function, x, and T_j gives an output mask, which is
    treated as a random binary vector. Here's a concrete example of the routing
    function: given output masks

    T_0 = [1, 0, 1, 0],
    T_1 = [1, 1, 0, 0]

    and an input x that causes the sparse linear transformation of x to be
    [0.3, −0.4, 0.2, 1.5], the routing function would yield outputs

    R(0, x) = [0.3, 0.0, 0.2, 0.0],
    R(1, x) = [0.3, −0.4, 0.0, 0.0].
    """

    def __init__(self, d_in, d_out, k, sparsity=0.7):
        """
        :param d_in: the number of dimensions in the input
        :type d_in: int
        :param d_out: the number of dimensions in the sparse linear output
        :type d_out: int
        :param k: the number of unique random binary vectors that can "route" the
        sparse linear output
        :type k: int
        :param sparsity: the sparsity in the SparseWeights layer (see
        nupic.torch.modules.SparseWeights for more details)
        :type sparsity: float
        """
        super(RoutingFunction, self).__init__()
        self.sparse_weights = SparseWeights(
            torch.nn.Linear(in_features=d_in, out_features=d_out, bias=False),
            sparsity=sparsity
        )
        self.output_masks = generate_random_binary_vectors(k, d_out)

    def forward(self, output_mask_inds, x):
        """
        Forward pass of the routing function

        :param output_mask_inds: a list of indices where the item at index j specifies
        the input to the routing function corresponding to batch item j in x
        :type output_mask_inds: list of int
        :param x: the batch input to the routing function
        :type x: torch Tensor
        """
        # TODO: remove this assertion, since this slows down forward computation
        assert len(output_mask_inds) == x.size(0), (
            "Length of output_mask_inds must match size of x on batch dimension 0"
        )

        mask = torch.stack([self.get_output_mask(j) for j in output_mask_inds], dim=0)
        output = self.sparse_weights(x)
        output = output * mask
        return output

    def get_output_mask(self, j):
        return self.output_masks[j, :]

    @property
    def num_output_masks(self):
        return self.output_masks.size(0)


if __name__ == "__main__":

    # The following code demonstrates how various output masks affect the output of the
    # routing function

    print("-- Routing example --")
    print("")

    input_dim = 20
    output_dim = 4
    k = 4
    batch_size = 1

    # Choose a random (input_dim)-dimensional input to the Routing Function
    x = torch.randn((1, 20))

    # Initialize RoutingFunction object with output_dim output dimensions and k output
    # masks
    R = RoutingFunction(d_in=input_dim, d_out=output_dim, k=k)

    # Print output masks
    for j in range(R.num_output_masks):
        print("output mask {}: {}".format(j, R.get_output_mask(j)))
    print("")

    # Print the output of the routing function with each of its output masks
    for j in range(R.num_output_masks):
        print("R({}, x): {}".format(j, R([j], x)))
    print("")
