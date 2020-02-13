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

import io
import itertools
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nupic.research.frameworks.pytorch.model_utils import (
    serialize_state_dict,
    set_random_seed,
)
from nupic.research.frameworks.pytorch.restore_utils import (
    get_linear_param_names,
    get_nonlinear_param_names,
    load_multi_state,
)
from nupic.torch.models import MNISTSparseCNN


def lower_forward(model, x):
    y = model.cnn1(x)
    y = model.cnn1_maxpool(y)
    y = model.cnn1_kwinner(y)
    y = model.cnn2(y)
    y = model.cnn2_maxpool(y)
    y = model.cnn2_kwinner(y)
    y = model.flatten(y)
    return y


def upper_forward(model, x):
    y = model.linear(x)
    y = model.linear_kwinner(y)
    y = model.output(y)
    y = model.softmax(y)
    return y


def full_forward(model, x):
    y = lower_forward(model, x)
    y = upper_forward(model, y)
    return y


class RestoreUtilsTest(unittest.TestCase):

    def setUp(self):

        set_random_seed(20)
        self.model = MNISTSparseCNN()
        self.model.eval()

        # Make all params twice as large to differentiate it from an init-ed model.
        for name, param in self.model.named_parameters():
            if ("cnn" in name or "linear" in name) and ("weight" in name):
                param[:] = param.data * 2

        # self.model.eval()
        self.in_1 = torch.rand(2, 1, 28, 28)
        self.in_2 = torch.rand(2, 1024)
        self.out_full = full_forward(self.model, self.in_1)
        self.out_lower = lower_forward(self.model, self.in_1)
        self.out_upper = upper_forward(self.model, self.in_2)

        # Create temporary results directory.
        self.tempdir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.tempdir.name) / Path("results")
        self.results_dir.mkdir()

        # Save model state.
        state = {}
        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.model.state_dict(), compresslevel=-1)
            state["model"] = buffer.getvalue()

        self.checkpoint_path = self.results_dir / Path("mymodel")
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(state, f)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_get_param_names(self):

        linear_params = get_linear_param_names(self.model)
        expected_linear_params = [
            "output.weight",
            "linear.module.bias",
            "output.bias",
            "linear.zero_weights",
            "linear.module.weight"
        ]
        self.assertTrue(set(linear_params) == set(expected_linear_params))

        nonlinear_params = get_nonlinear_param_names(self.model)
        expected_nonlinear_params = []
        for param, _ in itertools.chain(
            self.model.named_parameters(), self.model.named_buffers()
        ):
            if param not in expected_linear_params:
                expected_nonlinear_params.append(param)

        self.assertTrue(set(nonlinear_params) == set(expected_nonlinear_params))

    def test_load_full(self):

        # Initialize model with new random seed.
        set_random_seed(33)
        model = MNISTSparseCNN()
        model.eval()

        # Check output through the full network.
        for param1, param2 in zip(model.parameters(), self.model.parameters()):
            tot_eq = (param1 == param2).sum().item()
            self.assertNotEqual(tot_eq, np.prod(param1.shape))

        # Check output through the full network.
        out = full_forward(model, self.in_1)
        num_matches = out.isclose(self.out_full, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1)  # some correct

        # Check output through the lower network.
        out = lower_forward(model, self.in_1)
        num_matches = out.isclose(self.out_lower, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1337)  # some correct

        # Check output through the lower network.
        out = upper_forward(model, self.in_2)
        num_matches = out.isclose(self.out_upper, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1)  # some correct

        # Restore full model.
        model = load_multi_state(model, restore_full_model=self.checkpoint_path)
        model.eval()

        # Check output through the full network.
        for param1, param2 in zip(model.parameters(), self.model.parameters()):
            tot_eq = (param1 == param2).sum().item()
            self.assertEqual(tot_eq, np.prod(param1.shape))

        for buffer1, buffer2 in zip(model.buffers(), self.model.buffers()):
            tot_eq = (buffer1 == buffer2).sum().item()
            self.assertEqual(tot_eq, np.prod(buffer1.shape))

        out = full_forward(model, self.in_1)
        num_matches = out.isclose(self.out_full, atol=1e-2, rtol=0).sum().item()
        self.assertEqual(num_matches, 20)  # all correct

        # Check output through the lower network.
        out = lower_forward(model, self.in_1)
        num_matches = out.isclose(self.out_lower, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 2048)  # all correct

        # Check output through the lower network.
        out = upper_forward(model, self.in_2)
        num_matches = out.isclose(self.out_upper, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 20)  # all correct

    def test_load_nonlinear(self):

        # Initialize model with new random seed.
        set_random_seed(33)
        model = MNISTSparseCNN()
        model.eval()

        # Check output through the full network.
        for param1, param2 in zip(model.parameters(), self.model.parameters()):
            tot_eq = (param1 == param2).sum().item()
            self.assertNotEqual(tot_eq, np.prod(param1.shape))

        # Check output through the lower network.
        out = lower_forward(model, self.in_1)
        num_matches = out.isclose(self.out_lower, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1337)  # some correct

        # Check output through the lower network.
        out = upper_forward(model, self.in_2)
        num_matches = out.isclose(self.out_upper, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1)  # some correct

        # Restore full model.
        model = load_multi_state(model, restore_nonlinear=self.checkpoint_path)
        model.eval()

        # Check output through the lower network.
        out = lower_forward(model, self.in_1)
        num_matches = out.isclose(self.out_lower, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 2048)  # all correct

        # Check output through the lower network.
        out = upper_forward(model, self.in_2)
        num_matches = out.isclose(self.out_upper, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1)

    def test_load_linear(self):

        # Initialize model with new random seed.
        set_random_seed(33)
        model = MNISTSparseCNN()
        model.eval()

        # Check output through the full network.
        for param1, param2 in zip(model.parameters(), self.model.parameters()):
            tot_eq = (param1 == param2).sum().item()
            self.assertNotEqual(tot_eq, np.prod(param1.shape))

        # Check output through the lower network.
        out = lower_forward(model, self.in_1)
        num_matches = out.isclose(self.out_lower, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1337)  # some correct

        # Check output through the lower network.
        out = upper_forward(model, self.in_2)
        num_matches = out.isclose(self.out_upper, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1)  # some correct

        # Restore full model.
        model = load_multi_state(model, restore_linear=self.checkpoint_path)
        model.eval()

        # Check output through the lower network.
        out = lower_forward(model, self.in_1)
        num_matches = out.isclose(self.out_lower, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 1337)  # some correct

        # Check output through the lower network.
        out = upper_forward(model, self.in_2)
        num_matches = out.isclose(self.out_upper, atol=1e-2).sum().item()
        self.assertEqual(num_matches, 20)  # all correct


if __name__ == "__main__":
    unittest.main(verbosity=2)
