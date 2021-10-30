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

import pytest
import torch
from torchvision import transforms
from torchvision.datasets import FakeData

from nupic.research.frameworks.vernon import experiments, mixins

bp = pytest.importorskip(
    "nupic.research.backprop_structure.modules",
    reason="This test requires 'backprop_structure' framework package. "
           "Please install the packages using "
           "'pip install -e packages/backprop_structure'")

# Chosen so that all linear transformations can be of shape 4 x 4
INPUT_SHAPE = (1, 10, 10)


class SimpleVDropNet(torch.nn.Sequential):
    def __init__(self, input_shape=INPUT_SHAPE, hidden_size=100, output_size=10):
        super().__init__()

        self.vdrop_central_data = bp.MaskedVDropCentralData()
        self.flatten = torch.nn.Flatten()
        self.vdrop_conv2d = bp.VDropConv2d(1, 10, 3, self.vdrop_central_data)
        self.vdrop_linear_1 = bp.VDropLinear(640, hidden_size, self.vdrop_central_data)
        self.vdrop_linear_2 = bp.VDropLinear(
            hidden_size, output_size, self.vdrop_central_data
        )
        self.vdrop_central_data.finalize()

    def forward(self, x):
        self.vdrop_central_data.compute_forward_data()
        y = self.vdrop_conv2d(x)
        y = self.flatten(y)
        y = self.vdrop_linear_1(y)
        y = self.vdrop_linear_2(y)
        self.vdrop_central_data.clear_forward_data()
        return y


def fake_data(size=100, image_size=(1, 10, 10), train=False):
    return FakeData(size=size, image_size=image_size, transform=transforms.ToTensor())


class GlobalVDropSupervisedExperiment(
    mixins.PruneLowSNRGlobal, mixins.StepBasedLogging, experiments.SupervisedExperiment
):
    pass


simple_supervised_config = dict(
    experiment_class=GlobalVDropSupervisedExperiment,
    num_classes=10,
    # Dataset
    dataset_class=fake_data,
    # Number of epochs
    epochs=3,
    batch_size=10,
    batches_in_epoch=10,
    # Model class. Must inherit from "torch.nn.Module"
    model_class=SimpleVDropNet,
    # model model class arguments passed to the constructor
    prune_schedule=[(0, 0.9), (1, 0.8), (2, 0.3)],
    log_module_sparsities=True,
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
    # Suppress logging.
    log_level="NOTSET",
)


class GlobalSNRPruningTest(unittest.TestCase):
    def test_global_pruning(self):
        exp = simple_supervised_config["experiment_class"]()
        exp.setup_experiment(simple_supervised_config)
        # Loop through some pseudo epochs.
        exp.pre_epoch()
        for _ in range(exp.epochs):
            ret = exp.run_epoch()
        nonzero_params = ret["remaining_nonzero_parameters"]
        total_params = ret["total_prunable_parameters"]
        actual_density = nonzero_params / total_params
        desired_density = ret["model_density"]
        assert desired_density - 0.01 < actual_density < desired_density + 0.01


if __name__ == "__main__":
    unittest.main(verbosity=2)
