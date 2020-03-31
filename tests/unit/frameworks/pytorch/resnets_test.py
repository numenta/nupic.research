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

from nupic.research.frameworks.pytorch.modules import Mish
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.research.frameworks.pytorch.models.resnets import (
    ResNet,
    resnet18,
    resnet50,
    resnet101,
)
from nupic.research.frameworks.pytorch.sparse_layer_params import (
    LayerParams,
    SparseWeightsLayerParams,
)
from nupic.torch.modules import KWinners2d


class CustomKWinners2d(KWinners2d):
    def __init__(self, channels, percent_on=0.1, k_inference_factor=1.5,
                 boost_strength=1.0, boost_strength_factor=0.9, duty_cycle_period=1000,
                 local=False):
        super().__init__(channels, percent_on, k_inference_factor, boost_strength,
                         boost_strength_factor, duty_cycle_period, local)


def my_auto_sparse_activation_params(in_channels, out_channels, kernel_size):
    """
    A custom auto sparse params function.
    :return: a dict to pass to `KWinners2d` as params.
    """
    return dict(
        percent_on=0.25,
        boost_strength=10.0,
        boost_strength_factor=0.9,
        k_inference_factor=1.0,
    )


def my_auto_sparse_custom_activation_params(in_channels, out_channels, kernel_size):
    """
    A custom auto sparse params function.
    :return: a dict to pass to `KWinners2d` as params.
    """
    return dict(
        kwinner_class=CustomKWinners2d,
        percent_on=0.25,
        boost_strength=10.0,
        boost_strength_factor=0.9,
        k_inference_factor=1.0,
    )


def my_auto_sparse_conv_params(in_channels, out_channels, kernel_size):
    """
    Custom weight params.
    :return: a dict to pass to `SparseWeights2d`
    """
    return dict(
        weight_sparsity=0.42,
    )


def my_auto_sparse_linear_params(input_size, output_size):
    """
    Custom weight params.
    :return: a dict to pass to `SparseWeights`
    """
    return dict(
        weight_sparsity=0.42,
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

    def test_custom_kwinner_auto_params(self):
        """Create sparse ResNets with custom kwinner auto params."""

        default_kw_net = ResNet(
            config=dict(num_classes=10,
                        defaults_sparse=True,
                        activation_params_func=my_auto_sparse_activation_params,
                        conv_params_func=my_auto_sparse_conv_params)
        )
        custom_kw_net = ResNet(
            config=dict(num_classes=10,
                        defaults_sparse=True,
                        activation_params_func=my_auto_sparse_custom_activation_params,
                        conv_params_func=my_auto_sparse_conv_params)
        )

        default_kwinners = list(filter(lambda x: type(x) == KWinners2d,
                                       default_kw_net.modules()))
        custom_kwinners = list(filter(lambda x: type(x) == CustomKWinners2d,
                                      custom_kw_net.modules()))
        self.assertEqual(len(default_kwinners), 49)
        self.assertEqual(len(default_kwinners), len(custom_kwinners))

    def test_custom_auto_params(self):
        """Create sparse ResNets with custom auto params."""

        net = ResNet(
            config=dict(num_classes=10,
                        defaults_sparse=True,
                        activation_params_func=my_auto_sparse_activation_params,
                        conv_params_func=my_auto_sparse_conv_params)
        )
        net(Variable(torch.randn(2, 3, 32, 32)))

        params_sparse, nonzero_params_sparse = count_nonzero_params(net)
        self.assertAlmostEqual(float(nonzero_params_sparse) / params_sparse,
                               0.42, delta=0.01)

        self.assertIsInstance(net, ResNet, "Loads ResNet50 with custom auto params")

    def test_custom_auto_linear_params(self):

        # Using the default `layer_params_type`.
        net = ResNet(
            config=dict(num_classes=10,
                        defaults_sparse=False,  # -> dense convolutions
                        linear_params_func=my_auto_sparse_linear_params)
        )
        net(Variable(torch.randn(2, 3, 32, 32)))

        for name, param in net.named_parameters():
            if "classifier.module.weight" in name:
                total_params = param.data.numel()
                nonzero_params = param.data.nonzero().size(0)
                self.assertAlmostEqual(
                    nonzero_params / total_params, 0.42, places=3)

        # Using a custom `layer_params_type` (but otherwise the same test).
        net = ResNet(
            config=dict(num_classes=10,
                        defaults_sparse=False,  # -> dense convolutions
                        layer_params_type=SparseWeightsLayerParams,
                        layer_params_kwargs=dict(linear_weight_sparsity=0.42))
        )
        net(Variable(torch.randn(2, 3, 32, 32)))

        for name, param in net.named_parameters():
            if "classifier.module.weight" in name:
                total_params = param.data.numel()
                nonzero_params = param.data.nonzero().size(0)
                self.assertAlmostEqual(
                    nonzero_params / total_params, 0.42, places=3)

    def test_custom_per_group(self):
        """Evaluate ResNets customized per group"""

        custom_sparse_params = dict(
            stem=SparseWeightsLayerParams(),
            filters64=dict(
                conv1x1_1=SparseWeightsLayerParams(
                    percent_on=0.3,
                    boost_strength=1.2,
                    boost_strength_factor=1.0,
                    local=False,
                    weight_sparsity=0.3,
                ),
                conv3x3_2=SparseWeightsLayerParams(
                    percent_on=0.1,
                    boost_strength=1.2,
                    boost_strength_factor=1.0,
                    local=True,
                    weight_sparsity=0.1,
                ),
                conv1x1_3=SparseWeightsLayerParams(weight_sparsity=0.1),
                shortcut=SparseWeightsLayerParams(percent_on=0.4, weight_sparsity=0.4),
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
            linear=SparseWeightsLayerParams(weight_sparsity=0.5),
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
                    conv1x1_1=SparseWeightsLayerParams(
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

    def test_configurable_activation(self):
        """
        Test for configuring activation that precedes kwinners.
        """

        # Test failure case: none torch.nn.Module `base_activation`
        with self.assertRaises(
            AssertionError,
            msg="`base_activation` should be subclassed from torch.nn.Module"
        ):
            base_activation = lambda: "dummy object"  # only pass the type, not the constructed module
            ResNet(config=dict(
                depth=18,
                base_activation=base_activation
            ))

        # Test case with Mish activation.
        base_activation = Mish
        resnet = ResNet(config=dict(
            depth=18,
            base_activation=base_activation
        ))
        # print(resnet._modules.items())
        mish_layers = []
        for layer in resnet.modules():
            if isinstance(layer, Mish):
                mish_layers.append(layer)
        self.assertEqual(len(mish_layers), 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
