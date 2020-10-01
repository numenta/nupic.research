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
This module contains unit tests to ensure the RoutingFunction class correctly
implements the random routing function
"""

import unittest

import torch

from nupic.research.frameworks.dendrites.routing import RoutingFunction


class RoutingFunctionTest(unittest.TestCase):
    """
    Tests to check the correctness of the routing function R(j, x), as implemented by
    the RoutingFunction class
    """

    def test_function_output_satisfies_masks(self):
        dim_in = 100
        dim_out = 100
        num_output_masks = 10
        sparsity = 0.7

        r = RoutingFunction(
            d_in=dim_in,
            d_out=dim_out,
            k=num_output_masks,
            sparsity=sparsity
        )

        x = torch.randn((1, dim_in))

        for j in range(num_output_masks):

            output = r([j], x)
            self.assertEqual(output.ndim, 2)
            output = output.view(-1)

            output_mask_j = r.get_output_mask(j)
            for i in range(dim_out):
                if output_mask_j[i] == 0.0:
                    self.assertEqual(output[i], 0.0)

    def test_function_output(self):
        dim_in = 100
        dim_out = 100
        num_output_masks = 10
        sparsity = 0.7

        r = RoutingFunction(
            d_in=dim_in,
            d_out=dim_out,
            k=num_output_masks,
            sparsity=sparsity
        )

        x = torch.randn((1, dim_in))
        output = torch.matmul(r.weights, x.view(-1))

        for j in range(num_output_masks):

            output_mask_j = r.get_output_mask(j)
            expected_output = output_mask_j * output

            actual_output = r([j], x)
            self.assertEqual(actual_output.ndim, 2)
            actual_output = actual_output.view(-1)

            for i in range(dim_out):
                self.assertEqual(actual_output[i], expected_output[i])


if __name__ == "__main__":
    unittest.main()
