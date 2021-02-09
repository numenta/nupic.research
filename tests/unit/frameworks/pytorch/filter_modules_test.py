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
from nupic.research.frameworks.pytorch.models import OMLNetwork, resnets


class FilterParamsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_include_name(self):
        """
        Test use of `include_names`.
        """

        oml = OMLNetwork(num_classes=10)
        named_modules = filter_modules(oml, include_names=["adaptation.0"])
        self.assertTrue(len(named_modules) == 1)
        self.assertTrue("adaptation.0" in named_modules)
        self.assertIsInstance(named_modules["adaptation.0"], torch.nn.Linear)

    def test_name_not_included(self):
        """
        Test the case when a param name does not exist in the network.
        """

        oml = OMLNetwork(num_classes=10)

        named_params = filter_modules(oml, include_names=["adaptation.1"])
        self.assertTrue(len(named_params) == 0)

    def test_get_conv_modules_by_pattern_and_type(self):
        """
        Ensure `include_patterns` and `include_modules` yields the same result
        when they are meant to identify the same params.
        """
        oml = OMLNetwork(num_classes=10)

        include_even_numbers = ["representation.\\d*[02468]"]
        named_modules1 = filter_modules(oml, include_patterns=include_even_numbers)
        self.assertTrue(len(named_modules1) == 7)  # 7 convs and a flatten layer

        include_conv_and_flatten = [torch.nn.Conv2d, torch.nn.Flatten]
        named_modules2 = filter_modules(oml, include_modules=include_conv_and_flatten)
        self.assertTrue(len(named_modules2) == 7)

        names1 = list(named_modules1.keys())
        names2 = list(named_modules2.keys())
        self.assertTrue(names1 == names2)

    def test_filter_out_resnet_linear_params(self):
        """
        Filter out only the linear params of resnet.
        """
        resnet = resnets.resnet50(num_classes=10)
        named_modules = filter_modules(resnet, include_modules=[torch.nn.Linear])
        self.assertTrue(len(named_modules) == 1)
        self.assertTrue("classifier" in named_modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
