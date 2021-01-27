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

import numpy as np
import torch

from nupic.research.frameworks.pytorch.hooks import TrackSparsityHook
from nupic.torch.modules import KWinners, SparseWeights


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        in_features = np.prod(input_shape)
        self.flatten = torch.nn.Flatten()
        self.kwinners = KWinners(n=16, percent_on=0.75, k_inference_factor=1)
        self.classifier = SparseWeights(
            torch.nn.Linear(in_features, num_classes, bias=False),
            sparsity=0.5,
        )

    def forward(self, x):
        y = self.flatten(x)
        y = self.kwinners(y)
        return self.classifier(y)


def random_input(sparsity):
    """Random input of size (5, 64) with specified sparsity."""
    x = torch.rand(5, 64)
    off_count = int(x.shape[1] * sparsity)
    x[x == 0] = 1  # ensure there are no other zeros
    for i in range(5):
        rand_indices = np.random.choice(range(64), size=off_count, replace=False)
        x[i, rand_indices] = 0
    return x


class TrackSparsityHookTest(unittest.TestCase):

    def setUp(self):

        self.input_shape = (1, 4, 4)
        self.model = SimpleMLP(num_classes=10, input_shape=self.input_shape)

    def _test_single_forward_pass(self):

        x = torch.rand(self.input_shape)

        kwinners_hook = TrackSparsityHook(name="kwinners")
        self.model.kwinners.register_forward_hook(kwinners_hook)

        classifier_hook = TrackSparsityHook(name="classifier")
        self.model.classifier.register_forward_hook(classifier_hook)

        kwinners_hook.start_tracking()
        classifier_hook.start_tracking()
        self.model(x)

        _, output_sparsity = kwinners_hook.get_statistics()
        input_sparsity, _ = classifier_hook.get_statistics()

        self.assertEqual(input_sparsity, 0.25)
        self.assertEqual(output_sparsity, 0.25)

    def test_average_sparisty_over_mutliple_passes(self):
        """
        This tests ensures there's a cumulative average taken over multiple passes.
        """

        identity = torch.nn.Identity()

        hook = TrackSparsityHook(name="identity")
        identity.register_forward_hook(hook)
        hook.start_tracking()

        # First pass: 25% sparse
        x = random_input(sparsity=0.25)
        num_zeros = (x == 0).sum().item()
        self.assertEqual(num_zeros, 80)

        identity(x)
        input_sparsity, output_sparsity = hook.get_statistics()
        self.assertEqual(input_sparsity, output_sparsity)
        self.assertEqual(input_sparsity, 0.25)

        # Second pass: 50% sparse
        x = random_input(sparsity=0.50)
        num_zeros = (x == 0).sum().item()
        self.assertEqual(num_zeros, 160)

        identity(x)
        input_sparsity, output_sparsity = hook.get_statistics()
        self.assertEqual(input_sparsity, output_sparsity)
        self.assertEqual(input_sparsity, 0.375)  # cumulative average

        # Third pass: 75% sparse
        x = random_input(sparsity=0.75)
        num_zeros = (x == 0).sum().item()
        self.assertEqual(num_zeros, 240)

        identity(x)
        input_sparsity, output_sparsity = hook.get_statistics()
        self.assertEqual(input_sparsity, output_sparsity)
        self.assertEqual(input_sparsity, 0.5)  # cumulative average

    def test_restart_tracking(self):
        """
        This tests ensures tracked statistics get reset when restarting tracking.
        """

        relu = torch.nn.ReLU()

        hook = TrackSparsityHook(name="relu")
        relu.register_forward_hook(hook)
        hook.start_tracking()

        # First round: 25% sparse
        hook.start_tracking()
        x = random_input(sparsity=0.25)
        relu(x)
        x = random_input(sparsity=0.25)
        relu(x)
        input_sparsity, _ = hook.get_statistics()

        self.assertEqual(input_sparsity, 0.25)
        hook.stop_tracking()

        # Second round: 50% sparse
        hook.start_tracking()
        input_sparsity, output_sparsity = hook.get_statistics()
        self.assertEqual(input_sparsity, 0)
        self.assertEqual(input_sparsity, 0)

        x = random_input(sparsity=0.50)
        relu(x)
        x = random_input(sparsity=0.50)
        relu(x)
        input_sparsity2, _ = hook.get_statistics()

        self.assertEqual(input_sparsity2, 0.50)
        hook.stop_tracking()


if __name__ == "__main__":
    unittest.main(verbosity=2)
