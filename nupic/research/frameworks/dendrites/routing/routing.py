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

from random import randint

import torch
from torch.utils.data import Dataset

from nupic.torch.modules import SparseWeights

from .utils import generate_random_binary_vectors


class RoutingDataset(Dataset):
    """
    A dataset class for generating input-target pairs for the routing test specifically
    for a linear network with a single layer, where the inputs are random vectors
    sampled from U[-2, 2) each paired a binary sparse context vector
    """

    def __init__(
        self,
        routing_function,
        input_size,
        context_vectors,
        device,
        concat=False,
        dataset_size=1e4,
        x_min=-2.0,
        x_max=2.0,
    ):
        """
        :param routing_function: the random routing function
        :param input_size: the number of dimensions in the input to the routing
                           function and test module
        :param context_vectors: 2D torch Tensor in which each row gives a context
                                vector
        :param device: device to use ('cpu' or 'cuda')
        :param concat: if True, input and context vectors are concatenated together
        :param dataset_size: the number of (input, context, target) pairs that be
                             iterated over
        :param x_min: the minimum bound of the uniform distribution from which input
                      vectors are i.i.d. sampled along each input dimension
        :param x_max: the maximum bound of the uniform distribution from which input
                      vectors are i.i.d. sampled along each input dimension
        """
        super().__init__()
        self.function = routing_function
        self.num_output_masks = routing_function.num_output_masks
        self.input_size = input_size
        self.context_vectors = context_vectors
        self.device = device
        self.concat = concat
        self.size = int(dataset_size)

        # The following attributes are selected such that self.alpha * u + self.beta
        # gives a sample drawn from U[x_min, x_max) given a sample u ~ U[0, 1)
        self.alpha = (x_max - x_min)
        self.beta = x_min

    def __getitem__(self, idx):

        # To retrieve an input-context-target pair, first generate noise from a uniform
        # distribution as input (where the bounds of the distribution were specified in
        # the `__init__` method), and take the routing function's output on said input
        # using any of its output masks

        if idx > self.size:
            raise IndexError("Index {} is out of range".format(idx))
        torch.manual_seed(idx)

        x = self.alpha * torch.rand((self.input_size,)) - self.beta
        x = x.to(self.device)

        context_id = randint(0, self.num_output_masks - 1)
        context = self.context_vectors[context_id, :]
        context = context.to(self.device)

        target = self.function([context_id], x.view(1, -1))
        target = target.view(-1)

        if self.concat:
            x = torch.cat((x, context))
            return x, target

        return x, context, target

    def __len__(self):
        return self.size


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

    def __init__(self, d_in, d_out, k, device=None, sparsity=0.7):
        """
        :param d_in: the number of dimensions in the input
        :type d_in: int
        :param d_out: the number of dimensions in the sparse linear output
        :type d_out: int
        :param k: the number of unique random binary vectors that can "route" the
                  sparse linear output
        :param device: device to use ('cpu' or 'cuda')
        :type device: :class:`torch.device`
        :type k: int
        :param sparsity: the sparsity in the SparseWeights layer (see
                         nupic.torch.modules.SparseWeights for more details)
        :type sparsity: float
        """
        super().__init__()
        self.sparse_weights = SparseWeights(
            torch.nn.Linear(in_features=d_in, out_features=d_out, bias=False),
            sparsity=sparsity
        )
        self.output_masks = generate_random_binary_vectors(k, d_out)
        self.device = device if device is not None else torch.device("cpu")

    def forward(self, output_mask_inds, x):
        """
        Forward pass of the routing function

        :param output_mask_inds: a list of indices where the item at index j specifies
                                 the input to the routing function corresponding to
                                 batch item j in x
        :type output_mask_inds: list of int
        :param x: the batch input to the routing function
        :type x: torch Tensor
        """
        assert len(output_mask_inds) == x.size(0), (
            "Length of output_mask_inds must match size of x on batch dimension 0"
        )

        mask = torch.stack([self.get_output_mask(j) for j in output_mask_inds], dim=0)
        mask = mask.to(self.device)
        output = self.sparse_weights(x)
        output = output * mask
        return output

    def get_output_mask(self, j):
        return self.output_masks[j, :]

    @property
    def num_output_masks(self):
        return self.output_masks.size(0)

    @property
    def weights(self):
        return self.sparse_weights.module.weight.data


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
