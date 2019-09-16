#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from nupic.research.frameworks.dynamic_sparse.networks import MLPHeb
from nupic.torch.modules import KWinners


class CoactivationsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_coactivation_during_forward_pass(self):

        # network with 4 layers: input (3), hidden(4), hidden(5), output(2)
        network = MLPHeb(
            config=dict(
                input_size=3,
                num_classes=2,
                hidden_sizes=[4, 5],
                bias=False,
            )
        )
        network.init_hebbian()

        # setting same initial weights
        weights = [
            torch.randn((3, 4)) - 0.5,
            torch.randn((4, 5)) - 0.5,
            torch.randn((5, 2)) - 0.5,
        ]
        weights_iter = iter(weights)
        for m in network.modules():
            if isinstance(m, nn.Linear):
                # pytorch keeps weights tranposed
                m.weight.data = next(weights_iter).t()

        # ---------- Run one forward pass -----------

        inp = torch.randn((1, 3))

        # manually calculate coactivations:
        coactivations = []
        x = inp
        prev_act = x.clone()
        for idx, w in enumerate(weights):
            x = x @ w
            # apply ReLU in all layers, but the last
            if idx < len(weights) - 1:
                x = F.relu(x)
            # binarize and flat both x and prev act
            bin_prev_act = (prev_act > 0).float().view(-1)
            bin_x = (x > 0).float().view(-1)
            # compare with prev_act
            coactivations.append(torch.ger(bin_prev_act, bin_x))
            # reset prev_act
            prev_act = x.clone()

        network(inp.view(1, 1, 3))

        # compare coactivations
        matches = 0
        for ca_test, ca_network in zip(coactivations, network.coactivations):
            matches += int(torch.allclose(ca_test, ca_network))

        self.assertEqual(
            matches,
            len(coactivations),
            "Coactivation calculations should match for all layers after one pass",
        )

        # ---------- Run a batch -----------

        batch_size = 100
        batch_inp = torch.randn((batch_size, 1, 3))

        for i in range(100):
            inp = batch_inp[i, :, :]
            x = inp
            prev_act = x.clone()
            for idx, w in enumerate(weights):
                x = x @ w
                # apply ReLU in all layers, but the last
                if idx < len(weights) - 1:
                    x = F.relu(x)
                # binarize and flat both x and prev act
                bin_prev_act = (prev_act > 0).float().view(-1)
                bin_x = (x > 0).float().view(-1)
                # compare with prev_act
                coactivations[idx] += torch.ger(bin_prev_act, bin_x)
                # reset prev_act
                prev_act = x.clone()

        network(batch_inp)
        matches = 0
        for ca_test, ca_network in zip(coactivations, network.coactivations):
            # print(ca_test)
            # print(ca_network)
            matches += int(torch.allclose(ca_test, ca_network))

        self.assertEqual(
            matches,
            len(coactivations),
            "Coactivation calculations should match for after several passes",
        )

    def test_non_hebbian(self):
        network = MLPHeb(
            config=dict(
                input_size=5,
                num_classes=2,
                hidden_sizes=[4, 5],
                bias=False,
            )
        )
        inp = torch.randn((1, 5))
        network(inp.view(1, 1, 5))
        self.assertTrue(len(network.coactivations) == 0,
                        "Without init_hebbian it shouldn't compute coactivations.")

    def test_k_winner_construction(self):
        """Test that we can create k-winners independently in each layer."""
        network = MLPHeb(
            config=dict(
                input_size=5,
                num_classes=2,
                hidden_sizes=[4, 5, 6],
                percent_on_k_winner=[0.2, 0.5, 0.1],
                boost_strength=[1.4, 1.5, 1.6],
                boost_strength_factor=[0.7, 0.8, 0.9],
                bias=False,
            )
        )
        self.assertIsInstance(network.classifier[1][1], KWinners)
        self.assertIsInstance(network.classifier[2][1], nn.ReLU)
        self.assertIsInstance(network.classifier[3][1], KWinners)
        self.assertEqual(network.classifier[1][1].percent_on, 0.2)
        self.assertEqual(network.classifier[3][1].percent_on, 0.1)
        self.assertEqual(network.classifier[1][1].boost_strength, 1.4)
        self.assertEqual(network.classifier[3][1].boost_strength, 1.6)
        self.assertEqual(network.classifier[1][1].boost_strength_factor, 0.7)
        self.assertEqual(network.classifier[3][1].boost_strength_factor, 0.9)

    def test_coactivation_during_forward_pass_k_winner(self):
        """Test just the k-winner portion of the coactivation logic."""
        network = MLPHeb(
            config=dict(
                input_size=5,
                num_classes=2,
                hidden_sizes=[4],
                percent_on_k_winner=[0.25],
                bias=False,
            )
        )
        network.init_hebbian()

        # setting initial weights
        weights = [
            torch.tensor([
                [1, 0.0, 0, 0],
                [1, 1.1, 0, 0],
                [0, 1.0, 1, 0],
                [0, 0.0, 1, 1],
                [0, 0.0, 0, 1],
            ], dtype=torch.float),
            torch.randn((4, 2)) - 0.5,
        ]
        weights_iter = iter(weights)
        for m in network.modules():
            if isinstance(m, nn.Linear):
                # pytorch keeps weights tranposed
                m.weight.data = next(weights_iter).t()

        # ---------- Run forward pass - the first unit should win -----------

        coact1 = torch.tensor([[1., 0., 0., 0.],
                               [1., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 0.]])
        inp = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        network(inp.view(1, 1, 5))
        self.assertAlmostEqual(float((network.coactivations[0] - coact1).sum()), 0.0)

        # ---------- Run forward pass - the second unit should win -----------

        coact2 = torch.tensor([[1., 0., 0., 0.],
                               [1., 1., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 0., 0.],
                               [0., 0., 0., 0.]])
        inp = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]])
        network(inp.view(1, 1, 5))
        self.assertAlmostEqual(float((network.coactivations[0] - coact2).sum()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
