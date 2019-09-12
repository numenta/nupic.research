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
                use_kwinners=False,
                hebbian_learning=True,
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
                m.weight.data = next(weights_iter).T

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
