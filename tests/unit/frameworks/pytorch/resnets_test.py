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
from collections import defaultdict

import numpy as np
import torch
import torch.nn
from torch.autograd import Variable
from torchvision import models

from nupic.research.frameworks.pytorch.models.resnets import (
    LayerParams,
    NoactLayerParams,
    ResNet,
    resnet18,
    resnet50,
    resnet101,
)


class ResnetTest(unittest.TestCase):
    """Simple execution tests, not assertions"""

    def test_default(self):
        """Evaluate if ResNet50 initializes and runs"""
        net = resnet50(config=dict(num_classes=10))
        net(Variable(torch.randn(2, 3, 32, 32)))

        self.assertIsInstance(net, ResNet, "Loads ResNet50 with default parameters")

    def test_custom_per_group(self):
        """Evaluate ResNets customized per group"""

        custom_sparse_params = dict(
            stem=LayerParams(),
            filters64=dict(
                conv1x1_1=LayerParams(
                    percent_on=0.3,
                    boost_strength=1.2,
                    boost_strength_factor=1.0,
                    local=False,
                    weights_density=0.3,
                ),
                conv3x3_2=LayerParams(
                    percent_on=0.1,
                    boost_strength=1.2,
                    boost_strength_factor=1.0,
                    local=True,
                    weights_density=0.1,
                ),
                conv1x1_3=NoactLayerParams(weights_density=0.1),
                shortcut=LayerParams(percent_on=0.4, weights_density=0.4),
            ),
            filters128=dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            filters256=dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            filters512=dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=NoactLayerParams(),
                shortcut=LayerParams(),
            ),
            linear=NoactLayerParams(weights_density=0.5),
        )

        net = ResNet(
            config=dict(depth=50, num_classes=10, sparse_params=custom_sparse_params)
        )
        net(Variable(torch.randn(2, 3, 32, 32)))

        self.assertIsInstance(net, ResNet, "Loads ResNet50 customized per group")

    def test_fully_customized(self):
        """Evaluate if ResNet of different sizes initializes and runs"""

        custom_sparse_params = dict(
            stem=LayerParams(),
            filters64=[  # 3 blocks
                dict(
                    conv1x1_1=LayerParams(
                        percent_on=0.3,
                        boost_strength=1.2,
                        boost_strength_factor=1.0,
                        local=False,
                        weights_density=0.3,
                    ),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            filters128=[  # 4 blocks
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            filters256=[  # 6 blocks
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            filters512=[  # 3 blocks
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=NoactLayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            linear=NoactLayerParams(),
        )

        net = ResNet(
            config=dict(depth=50, num_classes=10, sparse_params=custom_sparse_params)
        )
        net(Variable(torch.randn(2, 3, 32, 32)))

        self.assertIsInstance(net, ResNet, "Loads ResNet50 fully customized")

    def test_different_sizes(self):
        """Evaluate if ResNet of different sizes initializes and runs"""

        # larger resnet
        net = resnet101()
        net(Variable(torch.randn(2, 3, 32, 32)))

        self.assertIsInstance(net, ResNet, "Loads ResNet101")

        # smaller resnet
        net = resnet18()
        net(Variable(torch.randn(2, 3, 32, 32)))

        self.assertIsInstance(net, ResNet, "Loads ResNet18")

    def test_version_15(self):
        """Evaluates if dense resnet-50 is similar to torchvision resnet-50"""

        nets = dict(torchvision=models.resnet50(), dense_nupic=resnet50())

        # assert number of parameters is the same, looking at state dict
        params_per_network = {}
        for name, net in nets.items():
            total_params = 0
            for v in net.state_dict().values():
                total_params += np.prod(v.shape)
            params_per_network[name] = total_params

        self.assertEqual(
            params_per_network["torchvision"],
            params_per_network["dense_nupic"],
            "Dense resnet-50 should have the same number of parameters as torchvision"
            " resnet-50, based on state dict",
        )

        # assert if 1-number of param, 2-param types and 3-layer name are the same
        nets_params = defaultdict(list)
        for name, network in nets.items():
            total_params, param_types, layer_names = [], [], []
            for m in network.modules():
                if hasattr(m, "weight"):
                    total_params.append(np.prod(m.weight.data.shape))
                    param_types.append(m.weight.data.dtype)
                    layer_names.append(m._get_name())
            nets_params[name] = [total_params, param_types, layer_names]

        # verify number of params is the same
        self.assertEqual(
            nets_params["torchvision"][0],
            nets_params["dense_nupic"][0],
            "Dense resnet-50 should have same number of parameters as torchvision"
            " resnet-50, based on modules",
        )
        # verify types of params is the same
        self.assertEqual(
            nets_params["torchvision"][1],
            nets_params["dense_nupic"][1],
            "Dense resnet-50 should have the same type of params as torchvision"
            " resnet-50",
        )
        # verify layer names are the same
        self.assertEqual(
            nets_params["torchvision"][2],
            nets_params["dense_nupic"][2],
            "Dense resnet-50 should have the same layer names as torchvision resnet-50",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
