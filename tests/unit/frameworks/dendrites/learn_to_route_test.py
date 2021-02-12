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

import copy
import unittest

import torch.nn.functional as F

from nupic.research.frameworks.dendrites import AbsoluteMaxGatingDendriticLayer
from nupic.research.frameworks.dendrites.routing import train_dendrite_model
from nupic.research.frameworks.dendrites.routing.learn import (
    init_dataloader,
    init_optimizer,
    init_test_scenario,
)


class LearnToRouteTest(unittest.TestCase):
    """
    This test suite tests whether dendrite layer weights are changing when learning the
    routing task
    """

    def test_forward_weights(self):
        """
        Feed-forward weights should not be modified when `mode == "dendrites"`, and
        should be modified when `mode == "all"`
        """

        for mode in ("dendrites", "all", "learn_context"):

            r, dendrite_layer, context_model, context_vectors, device = init_test_scenario(
                mode=mode,
                dim_in=100,
                dim_out=100,
                num_contexts=10,
                dim_context=100,
                dendrite_module=AbsoluteMaxGatingDendriticLayer
            )

            dataloader = init_dataloader(
                routing_function=r,
                context_vectors=context_vectors,
                device=device,
                batch_size=64,
                x_min=-2.0,
                x_max=2.0
            )

            optimizer = init_optimizer(mode=mode, layer=dendrite_layer, context_model=context_model)

            forward_weights_before = copy.deepcopy(
                dendrite_layer.module.weight.data
            )


            # Perform a single training epoch
            train_dendrite_model(
                model=dendrite_layer,
                context_model=context_model,
                loader=dataloader,
                optimizer=optimizer,
                device=device,
                criterion=F.l1_loss
            )

            forward_weights_after = copy.deepcopy(
                dendrite_layer.module.weight.data
            )

            expected = (forward_weights_before == forward_weights_after).all()

            # If training both feed-forward and dendrite weights, we expect the
            # dendrite weights to change
            if mode == "all" or "learn_context":
                expected = not expected

            self.assertTrue(expected)

    def test_dendrite_weights(self):
        """
        Dendrite weights should be modified both when `mode == "dendrites"` and when
        `mode == "all"`
        """

        for mode in ("dendrites", "all", "learn_context"):

            r, dendrite_layer, context_model, context_vectors, device = init_test_scenario(
                mode=mode,
                dim_in=100,
                dim_out=100,
                num_contexts=10,
                dim_context=100,
                dendrite_module=AbsoluteMaxGatingDendriticLayer
            )

            dataloader = init_dataloader(
                routing_function=r,
                context_vectors=context_vectors,
                device=device,
                batch_size=64,
                x_min=-2.0,
                x_max=2.0
            )

            optimizer = init_optimizer(mode=mode, layer=dendrite_layer, context_model=context_model)

            dendrite_weights_before = copy.deepcopy(
                dendrite_layer.segments.weights.data
            )

            # Perform a single training epoch
            train_dendrite_model(
                model=dendrite_layer,
                context_model=context_model,
                loader=dataloader,
                optimizer=optimizer,
                device=device,
                criterion=F.l1_loss
            )

            dendrite_weights_after = copy.deepcopy(
                dendrite_layer.segments.weights.data
            )

            expected = (dendrite_weights_before != dendrite_weights_after).any()
            self.assertTrue(expected)

    def test_context_model(self):
        """
        Context model should be learned only when `mode == "learn_context"`
        """
        for mode in ("dendrites", "all", "learn_context"):

            r, dendrite_layer, context_model, context_vectors, device = init_test_scenario(
                mode=mode,
                dim_in=100,
                dim_out=100,
                num_contexts=10,
                dim_context=100,
                dendrite_module=AbsoluteMaxGatingDendriticLayer
            )

            dataloader = init_dataloader(
                routing_function=r,
                context_vectors=context_vectors,
                device=device,
                batch_size=64,
                x_min=-2.0,
                x_max=2.0
            )

            optimizer = init_optimizer(mode=mode, layer=dendrite_layer, context_model=context_model)

            if mode != "learn_context":
                assert context_model is None
            else:
                context_model_weights_before = copy.deepcopy(context_model.linear1.weight.data)
                # Perform a single training epoch
                train_dendrite_model(
                    model=dendrite_layer,
                    context_model=context_model,
                    loader=dataloader,
                    optimizer=optimizer,
                    device=device,
                    criterion=F.l1_loss
                )

                context_model_weights_after = copy.deepcopy(context_model.linear1.weight.data)

                expected = (context_model_weights_before != context_model_weights_after).any()
                self.assertTrue(expected)

if __name__ == "__main__":
    unittest.main()
