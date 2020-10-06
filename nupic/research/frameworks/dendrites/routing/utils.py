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


def generate_random_binary_vectors(k, n_dim, sparsity_level=0.5):
    """
    Returns a Torch tensor of shape (k, n_dim) where each each entry is 1
    with probability (1 - sparsity_level), 0 otherwise

    :param k: the number of unique binary vectors
    :type k: int
    :param n_dim: the size of each binary vector
    :type n_dim: int
    :param sparsity_level: the expected level of sparsity of each binary vector
    :type n_dim: float
    """
    binary_vectors = torch.rand((k, n_dim))
    binary_vectors = torch.where(
        binary_vectors > sparsity_level,
        torch.zeros((k, n_dim)),
        torch.ones((k, n_dim))
    )
    return binary_vectors
