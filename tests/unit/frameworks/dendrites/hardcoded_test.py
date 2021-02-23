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

import unittest
from copy import deepcopy

import torch
from numpy.random import randint

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import (
    RoutingFunction,
    generate_context_vectors,
    get_gating_context_weights,
)


class HardcodedErrorTest(unittest.TestCase):
    """
    Tests the hand-picked context weights for performance on the hardcoded routing test
    in a dendritic network that uses dendrites for gating
    """

    def test_mean_abs_error(self):
        """
        Ensure mean absolute error retrieved by the hardcoded test is no larger than a
        chosen epsilon
        """

        # Set `epsilon` to a value that the mean absolute error between the routing
        # output and dendritic network (with hardcoded dendritic weights) output should
        # never exceed
        epsilon = 0.01

        # These hyperparameters control the size of the input and output to the routing
        # function (and dendritic network), the number of dendritic weights, the size
        # of the context vector, and batch size over which the mean absolute error is
        # computed
        dim_in = 100
        dim_out = 100
        num_contexts = 10
        dim_context = 100
        batch_size = 100

        r = RoutingFunction(dim_in=dim_in, dim_out=dim_out, k=num_contexts,
                            sparsity=0.7)

        context_vectors = generate_context_vectors(num_contexts=num_contexts,
                                                   n_dim=dim_context,
                                                   percent_on=0.2)

        module = AbsoluteMaxGatingDendriticLayer(module=r.sparse_weights.module,
                                                 num_segments=num_contexts,
                                                 dim_context=dim_context,
                                                 module_sparsity=0.7,
                                                 dendrite_sparsity=0.0)

        module.register_buffer("zero_mask",
                               deepcopy(r.sparse_weights.zero_mask.half()))

        hardcoded_weights = get_gating_context_weights(output_masks=r.output_masks,
                                                       context_vectors=context_vectors,
                                                       num_dendrites=num_contexts)
        module.segments.weights.data = hardcoded_weights

        x_test = 4.0 * torch.rand((batch_size, dim_in)) - 2.0  # sampled from U(-2, 2)
        context_inds_test = randint(low=0, high=num_contexts, size=batch_size).tolist()
        context_test = torch.stack(
            [context_vectors[j, :] for j in context_inds_test],
            dim=0
        )

        target = r(context_inds_test, x_test)
        actual = module(x_test, context_test)

        result = torch.abs(target - actual).mean().item()  # Mean absolute error
        self.assertLess(result, epsilon)


if __name__ == "__main__":
    unittest.main()
