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
import math

import numpy as np
import torch
import torch.nn as nn

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.torch.modules.sparse_weights import SparseWeights, SparseWeights2d


def select_random_blocks(num_blocks, num_zero_blocks, verbose=False):
    """
    Return a list of block indices that should be zero, and a list of block indices
    that should be non-zero.  The number of zero blocks and non-zero blocks total
    to num_blocks.
    """
    randomized_blocks = np.random.permutation(num_blocks)
    zero_blocks = randomized_blocks[:num_zero_blocks]
    non_zero_blocks = randomized_blocks[num_zero_blocks:]
    if verbose:
        print("zero blocks:", zero_blocks)
        print("non zero blocks:", non_zero_blocks)
    return zero_blocks, non_zero_blocks


def consolidated_zero_indices(input_size, percent_on, zero_blocks=None,
                              non_zero_blocks=None):
    """
    Return the indices of zero elements for one linear unit. Ensure that we have a
    large number of runs of 64 elements with all zeros.

    TODO: Currently this only works if input_size is a multiple of 64.
    """
    num_blocks = math.ceil(input_size / 64.0)

    # Randomly select which blocks are going to be zero and non-zero
    if zero_blocks is None:
        num_zero_blocks = int(num_blocks - (3 * percent_on * num_blocks))
        zero_blocks, non_zero_blocks = select_random_blocks(num_blocks, num_zero_blocks)
    else:
        num_zero_blocks = len(zero_blocks)
    num_nonzero_blocks = num_blocks - num_zero_blocks

    num_zero_bits_in_nonzero_blocks = num_nonzero_blocks * 64 - round(
        percent_on * input_size)

    indices_in_zero_blocks = [b * 64 + off for b in zero_blocks for off in range(64)]
    indices_in_nonzero_blocks = [b * 64 + off
                                 for b in non_zero_blocks for off in range(64)]
    zero_indices_in_nonzero_blocks = np.random.permutation(
        indices_in_nonzero_blocks)[:num_zero_bits_in_nonzero_blocks]

    indices = indices_in_zero_blocks + list(zero_indices_in_nonzero_blocks)
    indices.sort()
    return np.array(indices)


class ConsolidatedSparseWeights(SparseWeights):
    def __init__(self, module, weight_sparsity):
        """Enforce somewhat blocky weight sparsity on linear module during training.

        Sample usage:

          model = nn.Linear(784, 10)
          model = SparseWeights(model, 0.4)

        :param module:
          The module to sparsify the weights
        :param weight_sparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(ConsolidatedSparseWeights, self).__init__(module, weight_sparsity)
        assert isinstance(module, nn.Linear)

    def compute_indices(self):
        # print("In ConsolidatedSparseWeights linear")
        # For each unit, decide which weights are going to be zero
        output_size, input_size = self.module.weight.shape
        num_zeros = int(round((1.0 - self.weight_sparsity) * input_size))

        output_indices = np.arange(output_size)
        input_indices = np.array(
            [consolidated_zero_indices(input_size, self.weight_sparsity)
             for _ in output_indices],
            dtype=np.long,
        )

        # Create tensor indices for all non-zero weights
        zero_indices = np.empty((output_size, num_zeros, 2), dtype=np.long)
        zero_indices[:, :, 0] = output_indices[:, None]
        zero_indices[:, :, 1] = input_indices
        zero_indices = zero_indices.reshape(-1, 2)
        return torch.from_numpy(zero_indices.transpose())


class ConsolidatedSparseWeights2D(SparseWeights2d):
    def __init__(self, module, weight_sparsity):
        """Enforce somewhat blocky weight sparsity on CNN modules Sample usage:

          model = nn.Conv2d(in_channels, out_channels, kernel_size, ...)
          model = SparseWeights2d(model, 0.4)

        :param module:
          The module to sparsify the weights
        :param weight_sparsity:
          Pct of weights that are allowed to be non-zero in the layer.
        """
        super(ConsolidatedSparseWeights2D, self).__init__(module, weight_sparsity)
        assert isinstance(module, nn.Conv2d)

    def compute_indices(self):
        # print("In ConsolidatedSparseWeights Conv2d")
        # For each unit, decide which weights are going to be zero
        in_channels = self.module.in_channels
        out_channels = self.module.out_channels
        kernel_size = self.module.kernel_size

        input_size = in_channels * kernel_size[0] * kernel_size[1]
        num_zeros = int(round((1.0 - self.weight_sparsity) * input_size))

        # Store a set of out_channels/4 blocks.
        num_blocks = math.ceil(input_size / 64.0)
        num_zero_blocks = int(num_blocks - (self.weight_sparsity * num_blocks + 2))
        output_indices = np.arange(out_channels / 4)

        sparse_blocks = [
            select_random_blocks(num_blocks, num_zero_blocks, False)
            for _ in output_indices
        ]
        output_indices = np.arange(out_channels)
        input_indices = np.array(
            [consolidated_zero_indices(input_size, self.weight_sparsity,
                                       zero_blocks=sparse_blocks[int(i / 4)][0],
                                       non_zero_blocks=sparse_blocks[int(i / 4)][1]
                                       )
             for i in output_indices],
            dtype=np.long,
        )

        # Create tensor indices for all non-zero weights
        zero_indices = np.empty((out_channels, num_zeros, 2), dtype=np.long)
        zero_indices[:, :, 0] = output_indices[:, None]
        zero_indices[:, :, 1] = input_indices

        # for i in range(10):
        #     for channel in range(0, 8):
        #         print("channel ", channel, "block", i)
        #         for j in range(i*64, (i+1)*64):
        #             print(zero_indices[channel][j], end=" "),
        #         print()
        zero_indices = zero_indices.reshape(-1, 2)

        return torch.from_numpy(zero_indices.transpose())

    def rezero_weights(self):
        zero_idx = (self.zero_weights[0], self.zero_weights[1])
        self.module.weight.data.view(self.module.out_channels, -1)[zero_idx] = 0.0


if __name__ == "__main__":
    # inds = consolidated_zero_indices(1600, 0.05)

    # model = torch.nn.Sequential(
    #     ConsolidatedSparseWeights(torch.nn.Linear(1600, 1500), 0.05),
    # )
    # print("Number of non-zero weights:", count_nonzero_params(model))

    # inds = consolidated_zero_indices(64*5*5, 0.1)
    model2 = torch.nn.Sequential(
        ConsolidatedSparseWeights2D(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=0,
                stride=1,
            ), 0.1)
    )
    print("Number of non-zero weights:", count_nonzero_params(model2))
