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

from nupic.research.frameworks.dendrites import AbsoluateMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import get_gating_context_weights
from nupic.research.frameworks.dendrites.routing.hardcoded import (
    run_hardcoded_routing_test,
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
        d_in = 100
        d_out = 100
        num_contexts = 10
        d_context = 100
        batch_size = 100

        result = run_hardcoded_routing_test(
            d_in=d_in,
            d_out=d_out,
            k=num_contexts,
            d_context=d_context,
            dendrite_module=AbsoluateMaxGatingDendriticLayer,
            context_weights_fn=get_gating_context_weights,
            batch_size=batch_size
        )

        mean_abs_error = result["mean_abs_error"]
        self.assertLess(mean_abs_error, epsilon)


if __name__ == "__main__":
    unittest.main()
