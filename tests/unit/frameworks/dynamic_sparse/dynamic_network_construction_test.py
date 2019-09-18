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
from torch import nn

from nupic.research.frameworks.dynamic_sparse.networks import (
    DSConv2d,
    DSLinear,
    GSCSparseCNN,
    init_coactivation_tracking,
    make_dsnn,
    squash_layers,
    swap_layers,
)
from nupic.torch.modules import KWinners, KWinners2d, SparseWeights


class NetworkConstructionTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_dynamic_gsc(self):

        config = dict(
            prune_methods=[None, "dynamic-conv", "dynamic-linear", None]
        )

        net = GSCSparseCNN()
        make_dsnn(net, config)

        dynamic_layers = []
        for l in net.modules():
            if isinstance(l, (DSLinear, DSConv2d)):
                dynamic_layers.append(l)

        assert len(dynamic_layers) == 2
        assert isinstance(dynamic_layers[0], DSConv2d)
        assert isinstance(dynamic_layers[1], DSLinear)

        assert len(net) == 14

        # Swap activation with max pool
        net = swap_layers(net, nn.MaxPool2d, KWinners2d)
        assert len(net) == 14

        # Squash dynamic conv layer with its activation.
        net = squash_layers(
            net, DSConv2d, nn.BatchNorm2d, KWinners2d, transfer_forward_hook=True)
        assert len(net) == 12

        # Squash linear conv layer with its activation.
        net = squash_layers(
            net, SparseWeights, nn.BatchNorm1d, KWinners, transfer_forward_hook=True)
        assert len(net) == 10

        # Exercise forward pass.
        net.apply(init_coactivation_tracking)
        input_tensor = torch.rand(2, 1, 32, 32)
        net(input_tensor)

        # Ensure coactivations have been updated.
        conv_coacts = net[4][0].coactivations
        lin_coacts = net[7][0].module.coactivations
        assert not conv_coacts.allclose(torch.zeros_like(conv_coacts))
        assert not lin_coacts.allclose(torch.zeros_like(lin_coacts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
