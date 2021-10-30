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

import torch

from nupic.research.frameworks.pytorch.model_utils import filter_params
from nupic.research.frameworks.pytorch.models import resnets


class FilterParamsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_include_name(self):
        """
        Test use of `include_names`.
        """

        resnet = resnets.resnet50(num_classes=10)
        named_params = filter_params(resnet, include_names=["classifier.weight"])
        self.assertEqual(len(named_params), 1)
        self.assertIn("classifier.weight", named_params)
        self.assertEqual(named_params["classifier.weight"].shape, (10, 2048))

    def test_name_not_included(self):
        """
        Test the case when a param name does not exist in the network.
        """
        resnet = resnets.resnet50(num_classes=10)
        named_params = filter_params(resnet, include_names=["classifier"])
        self.assertEqual(len(named_params), 0)

    def test_get_conv_modules_by_pattern_and_type(self):
        """
        Ensure `include_patterns` and `include_modules` yields the same result
        when they are meant to identify the same params.
        """
        resnet = resnets.resnet50(num_classes=10)
        named_params1 = filter_params(
            resnet, include_patterns=["classifier"])
        self.assertEqual(len(named_params1), 2)  # linear weight + bias

        named_params2 = filter_params(
            resnet, include_modules=[torch.nn.Linear])
        self.assertEqual(len(named_params2), 2)

        names1 = list(named_params1.keys())
        names2 = list(named_params2.keys())
        self.assertEqual(names1, names2)

    def test_filter_out_resnet_linear_params(self):
        """
        Filter out only the linear params of resnet.
        """
        resnet = resnets.resnet50(num_classes=10)
        named_params = filter_params(resnet, include_modules=[torch.nn.Linear])
        self.assertEqual(len(named_params), 2)
        self.assertIn("classifier.weight", named_params)
        self.assertIn("classifier.bias", named_params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
