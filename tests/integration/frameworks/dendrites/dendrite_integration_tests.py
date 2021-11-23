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
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

dendrites = pytest.importorskip("nupic.research.frameworks.dendrites")
dendrites_mixins = pytest.importorskip("nupic.research.frameworks.dendrites.mixins")
dendrites_experiments = pytest.importorskip(
    "nupic.research.frameworks.dendrites.dendrite_cl_experiment"
)

vernon_mixins = pytest.importorskip("nupic.research.frameworks.vernon.mixins")
cl_mixins = pytest.importorskip("nupic.research.frameworks.continual_learning.mixins")


class LocalExperiment(
    vernon_mixins.RezeroWeights,
    cl_mixins.SynapticIntelligence,
    dendrites_mixins.PrototypeContext,
    dendrites_experiments.DendriteContinualLearningExperiment,
):
    # computes error loss with and without SI and make sure the
    # SI regularization term is being used
    def error_loss(self, output, target, reduction="mean"):

        # computes dendrites loss without SI
        dendrites_original_loss = super(
            dendrites_experiments.DendriteContinualLearningExperiment, self
        ).error_loss(output, target, reduction)

        # computes dendrites loss with SI
        dendrites_with_si_loss = super().error_loss(output, target, reduction)
        diff = dendrites_with_si_loss - dendrites_original_loss

        # Computes surrogate loss
        regularization_term = torch.tensor(0.0, device=self.device)

        if self.current_task > 1:
            for name, param in self.named_si_parameters():

                big_omega = self.big_omega[name]
                old_param = self.stable_params[name]

                regularization_term += (big_omega * ((param - old_param) ** 2)).sum()

        regularization_term = self.c * regularization_term

        assert diff == regularization_term

        return dendrites_with_si_loss


def fake_data(size=100, image_size=(1, 28, 28), train=False):
    return FakeData(size=size, image_size=image_size, transform=ToTensor())


# A relatively quick running experiment for debugging
integration_test_config = dict(
    experiment_class=LocalExperiment,
    num_classes=10,
    # Dataset
    dataset_class=fake_data,
    # Number of epochs
    epochs=1,
    batch_size=10,
    # Model class. Must inherit from "torch.nn.Module"
    model_class=dendrites.DendriticMLP,
    model_args=dict(
        input_size=784,
        output_size=10,
        hidden_sizes=[2000, 2000],  # Note we use 2000 hidden units instead of 2048 for
        # a better comparison with SI and XdG
        num_segments=10,
        dim_context=784,
        kw=True,
        kw_percent_on=0.05,
        dendrite_weight_sparsity=0.0,
        weight_sparsity=0.5,
        context_percent_on=0.1,
    ),
    # Optimizer class class arguments passed to the constructor
    optimizer_args=dict(lr=0.1),
    fast_params=[".*"],  # <- all params get updated in inner loop
    # Suppress logging.
    log_level="NOTSET",
    si_args=dict(
        c=0.1,
        damping=0.1,
    ),
)


class DendriteIntegrationTest(unittest.TestCase):
    """
    This is a test class for testing Dendrite integration
    with Vernon and Continual Learning
    """

    def test_si_mixin(self):
        """
        Tests Synaptic Intelligence mixins integrations.
        """

        # Setup experiment and initialize model.
        exp = integration_test_config["experiment_class"]()
        exp.setup_experiment(integration_test_config)

        # Checks if config hyper params are as defined
        self.assertTrue(exp.c == 0.1)
        self.assertTrue(exp.damping == 0.1)
        self.assertFalse(exp.apply_to_dendrites)

        # Checks if the modules are correctly working together
        exp.run_iteration()

        # Still don't know how to check the results


if __name__ == "__main__":
    unittest.main(verbosity=2)
