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
import torch.nn
from torch.autograd import Variable

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.research.frameworks.pytorch.models.resnets import (
    ResNet,
    resnet18,
    resnet50,
    resnet101,
)
from nupic.research.frameworks.pytorch.sparse_layer_params import (
    LayerParams,
    SpareWeightsLayerParams,
)


class ResnetTest(unittest.TestCase):
    """Simple execution tests, not assertions"""

    def test_default(self):
        """Evaluate if ResNet50 initializes and runs"""
        net = resnet50(config=dict(num_classes=10))
        net(Variable(torch.randn(2, 3, 32, 32)))

        self.assertIsInstance(net, ResNet, "Loads ResNet50 with default parameters")

    def test_default_sparse(self):
        """Create the default sparse network"""
        net = resnet50(config=dict(num_classes=10, defaults_sparse=True))
        net(Variable(torch.randn(2, 3, 32, 32)))
        self.assertIsInstance(net, ResNet, "ResNet50 with default sparse parameters")

        # Test on CUDA if available
        if torch.cuda.is_available():
            net.cuda()
            x = Variable(torch.randn(16, 3, 224, 224))
            net(x.cuda())

    def test_params_count(self):
        """
        Test the number of non-zero parameters for default dense and sparse networks
        """
        dense_net = resnet50(config=dict(num_classes=10))
        dense_net(Variable(torch.randn(2, 3, 32, 32)))

        sparse_net = resnet50(config=dict(num_classes=10, defaults_sparse=True))
        sparse_net(Variable(torch.randn(2, 3, 32, 32)))

        total_params_dense, total_nonzero_params_dense = count_nonzero_params(dense_net)
        self.assertGreater(total_params_dense, 23500000)
        self.assertGreaterEqual(total_params_dense, total_nonzero_params_dense)

        params_sparse, nonzero_params_sparse = count_nonzero_params(sparse_net)

        self.assertEqual(params_sparse, total_params_dense)
        self.assertLess(nonzero_params_sparse, 10000000)

    def test_custom_per_group(self):
        """Evaluate ResNets customized per group"""

        custom_sparse_params = dict(
            stem=SpareWeightsLayerParams(),
            filters64=dict(
                conv1x1_1=SpareWeightsLayerParams(
                    percent_on=0.3,
                    boost_strength=1.2,
                    boost_strength_factor=1.0,
                    local=False,
                    weight_sparsity=0.3,
                ),
                conv3x3_2=SpareWeightsLayerParams(
                    percent_on=0.1,
                    boost_strength=1.2,
                    boost_strength_factor=1.0,
                    local=True,
                    weight_sparsity=0.1,
                ),
                conv1x1_3=SpareWeightsLayerParams(weight_sparsity=0.1),
                shortcut=SpareWeightsLayerParams(percent_on=0.4, weight_sparsity=0.4),
            ),
            filters128=dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=LayerParams(),
                shortcut=LayerParams(),
            ),
            filters256=dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=LayerParams(),
                shortcut=LayerParams(),
            ),
            filters512=dict(
                conv1x1_1=LayerParams(),
                conv3x3_2=LayerParams(),
                conv1x1_3=LayerParams(),
                shortcut=LayerParams(),
            ),
            linear=SpareWeightsLayerParams(weight_sparsity=0.5),
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
                    conv1x1_1=SpareWeightsLayerParams(
                        percent_on=0.3,
                        boost_strength=1.2,
                        boost_strength_factor=1.0,
                        local=False,
                        weight_sparsity=0.3,
                    ),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            filters128=[  # 4 blocks
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            filters256=[  # 6 blocks
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            filters512=[  # 3 blocks
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
                dict(
                    conv1x1_1=LayerParams(),
                    conv3x3_2=LayerParams(),
                    conv1x1_3=LayerParams(),
                    shortcut=LayerParams(),
                ),
            ],
            linear=LayerParams(),
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
