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

import nupic.research.frameworks.dendrites.functional as F
from nupic.research.frameworks.dendrites import (
    DendriticAbsoluteMaxGate1d,
    DendriticAbsoluteMaxGate2d,
    DendriticBias1d,
    DendriticGate1d,
    DendriticGate2d,
)


class ApplyDendritesModulesTest(unittest.TestCase):

    def setUp(self):
        batch_size = 5
        num_units = 4
        num_segments = 10
        channels = 3  # 2d versions will use channels x num_units x num_units

        self.y = torch.rand(batch_size, num_units)
        self.dendrite_activations = torch.rand(batch_size, num_units, num_segments)

        self.y_2d = torch.rand(batch_size, channels, num_units, num_units)
        self.dendrite_activations_2d = torch.rand(batch_size, channels, num_units)

    def test_dendritic_bias(self):
        """
        Ensure `dendritic_bias_1d` and `DendriticBias1d` yield the same outputs.
        """
        module = DendriticBias1d()
        output_a = F.dendritic_bias_1d(self.y, self.dendrite_activations)
        output_b = module(self.y, self.dendrite_activations)
        torch.testing.assert_close(output_a[0], output_b[0])

    def test_dendritic_gate(self):
        """
        Ensure `dendritic_gate_1d` and `DendriticGate1d` yield the same outputs.
        """
        module = DendriticGate1d()
        output_a = F.dendritic_gate_1d(self.y, self.dendrite_activations)
        output_b = module(self.y, self.dendrite_activations)
        torch.testing.assert_close(output_a[0], output_b[0])

    def test_dendritic_absolute_max_gate(self):
        """
        Ensure `dendritic_absolute_max_gate_1d` and `DendriticAbsoluteMaxGate1d` yield
        the same outputs.
        """
        module = DendriticAbsoluteMaxGate1d()
        output_a = F.dendritic_absolute_max_gate_1d(self.y, self.dendrite_activations)
        output_b = module(self.y, self.dendrite_activations)
        torch.testing.assert_close(output_a[0], output_b[0])

    def test_dendritic_gate_2d(self):
        """
        Ensure `dendritic_gate_2d` and `DendriticGate2d` yield the same outputs.
        """
        module = DendriticGate2d()
        output_a = F.dendritic_gate_2d(self.y_2d, self.dendrite_activations_2d)
        output_b = module(self.y_2d, self.dendrite_activations_2d)
        torch.testing.assert_close(output_a[0], output_b[0])

    def test_dendritic_absolute_max_gate_2d(self):
        """
        Ensure `dendritic_absolute_max_gate_2d` and `DendriticAbsoluteMaxGate2d` yield
        the same outputs.
        """
        module = DendriticAbsoluteMaxGate2d()
        output_a = F.dendritic_absolute_max_gate_2d(self.y_2d,
                                                    self.dendrite_activations_2d)
        output_b = module(self.y_2d, self.dendrite_activations_2d)
        torch.testing.assert_close(output_a[0], output_b[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
