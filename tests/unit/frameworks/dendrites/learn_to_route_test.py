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

import torch.nn.functional as F

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import train_dendrite_model
from nupic.research.frameworks.dendrites.routing.learned import (
    init_dataloader,
    init_optimizer,
    init_test_scenario,
)


class LearnToRouteTest(unittest.TestCase):
    """
    This test suite only applies to the "learning to route" scenario in which only
    dendritic weights are learned while feed-forward weights are fixed
    """

    def test_non_dendritic_weights(self):
        """ Non-dendritic weights should not be modified """

        r, dendritic_network, context_vectors, device = init_test_scenario(
            dim_in=100,
            dim_out=100,
            num_contexts=10,
            dim_context=100,
            dendrite_module=AbsoluteMaxGatingDendriticLayer
        )

        routing_test_dataloader = init_dataloader(
            routing_function=r,
            context_vectors=context_vectors,
            device=device,
            batch_size=64
        )

        optimizer = init_optimizer(network=dendritic_network)

        non_dendritic_weights_before = dendritic_network.module.weight.data

        # Perform a single training epoch
        train_dendrite_model(
            model=dendritic_network,
            loader=routing_test_dataloader,
            optimizer=optimizer,
            device=device,
            criterion=F.mse_loss
        )

        non_dendritic_weights_after = dendritic_network.module.weight.data

        expected = (non_dendritic_weights_before == non_dendritic_weights_after).all()
        self.assertTrue(expected)


if __name__ == "__main__":
    unittest.main()
