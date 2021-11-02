# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import torch

from nupic.research.frameworks.pytorch.model_utils import filter_modules
from nupic.research.frameworks.pytorch.models import resnets


class FilterParamsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_include_name(self):
        """
        Test use of `include_names`.
        """

        resnet = resnets.resnet50(num_classes=10)
        named_modules = filter_modules(resnet, include_names=["classifier"])
        self.assertEqual(len(named_modules), 1)
        self.assertIn("classifier", named_modules)
        self.assertIsInstance(named_modules["classifier"], torch.nn.Linear)

    def test_name_not_included(self):
        """
        Test the case when a param name does not exist in the network.
        """

        resnet = resnets.resnet50(num_classes=10)
        named_params = filter_modules(resnet, include_names=["adaptation.1"])
        self.assertEqual(len(named_params), 0)

    def test_get_conv_modules_by_pattern_and_type(self):
        """
        Ensure `include_patterns` and `include_modules` yields the same result
        when they are meant to identify the same params.
        """
        resnet = resnets.resnet50(num_classes=10)

        include_pooling_layers = ["features\\..*pool.*"]
        named_modules1 = filter_modules(resnet, include_patterns=include_pooling_layers)
        self.assertEqual(len(named_modules1), 2)

        pooling_layers_types = [
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            torch.nn.modules.pooling.MaxPool2d,
        ]
        named_modules2 = filter_modules(
            resnet, include_modules=pooling_layers_types)
        self.assertEqual(len(named_modules2), 2)

        names1 = list(named_modules1.keys())
        names2 = list(named_modules2.keys())
        self.assertEqual(names1, names2)

    def test_filter_out_resnet_linear_params(self):
        """
        Filter out only the linear params of resnet.
        """
        resnet = resnets.resnet50(num_classes=10)
        named_modules = filter_modules(resnet, include_modules=[torch.nn.Linear])
        self.assertEqual(len(named_modules), 1)
        self.assertIn("classifier", named_modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
