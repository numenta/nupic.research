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
# %%
import unittest

import numpy as np
import torch
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from nupic.research.frameworks.vernon import SupervisedExperiment, mixins


class SupervisedMultiCycleLR(mixins.MultiCycleLR, SupervisedExperiment):
    pass


class SimpleMLP(torch.nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        in_features = np.prod(input_shape)
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(in_features, num_classes, bias=False)

    def forward(self, x):
        return self.classifier(self.flatten(x))


def fake_data(size=100, image_size=(1, 4, 4), train=False):
    return FakeData(size=size, image_size=image_size, transform=ToTensor())


supervised_experiment = dict(
    experiment_class=SupervisedMultiCycleLR,
    num_classes=10,
    # Dataset
    dataset_class=fake_data,
    # Number of epochs
    epochs=10,
    batch_size=5,
    # Model class. Must inherit from "torch.nn.Module"
    model_class=SimpleMLP,
    # model model class arguments passed to the constructor
    model_args=dict(
        num_classes=10,
        input_shape=(1, 4, 4),
    ),
    # Optimizer class arguments, include multi_cycle_lr test args
    # (Not a great config, just testing different values for available args)
    optimizer_args=dict(lr=0.1),
    multi_cycle_lr_args=(
        (
            0,
            dict(
                max_lr=1.5,
                pct_start=0.2,
                anneal_strategy="linear",
                base_momentum=0.6,
                max_momentum=0.75,
                cycle_momentum=True,
                div_factor=6.0,
                final_div_factor=1000.0,
            ),
        ),
        (
            1,
            dict(
                max_lr=1.0,
                pct_start=0.2,
                anneal_strategy="cos",
                base_momentum=0.55,
                max_momentum=0.7,
                cycle_momentum=True,
                div_factor=3.0,
                final_div_factor=1000.0,
            ),
        ),
        (
            5,
            dict(
                max_lr=0.05,
                pct_start=0.2,
                anneal_strategy="linear",
                base_momentum=0.6,
                max_momentum=0.75,
                cycle_momentum=False,
                div_factor=25.0,
                final_div_factor=100.0,
            ),
        ),
    ),
)


class MultiCycleLRTest(unittest.TestCase):
    def test_supervised_experiment_with_cycle_lr(self):

        # Setup experiment and initialize model.
        exp = supervised_experiment["experiment_class"]()
        exp.setup_experiment(supervised_experiment)

        num_images = len(exp.train_loader.dataset)
        num_batches = -(-num_images // exp.train_loader.batch_size)

        # Loop through some pseudo epochs.
        cycle = 0
        for i in range(exp.epochs):
            lr = []

            exp.pre_epoch()
            # Loop through batches and get corresponding lr.
            for batch in range(num_batches):
                exp.pre_batch(exp.model, batch)
                exp.optimizer.step()
                lr.append(exp.get_lr()[0])
                exp.post_batch(
                    exp.model, 0, None, batch, exp.train_loader.batch_size, ""
                )
            exp.post_epoch()
            exp.current_epoch += 1

            # Get the current max lr according to the multiCycleLR config
            if (
                cycle < len(supervised_experiment["multi_cycle_lr_args"])
                and supervised_experiment["multi_cycle_lr_args"][cycle][0] == i
            ):
                current_lr_conf = supervised_experiment["multi_cycle_lr_args"][
                    cycle
                ][1]
                cycle += 1
                # If new cycle is started, check that initial lr is
                # max_lr/div_factor
                initial_lr = (
                    current_lr_conf["max_lr"] / current_lr_conf["div_factor"]
                )
                self.assertAlmostEqual(lr[0], initial_lr)

            # Check that lr of this epoch stays below max lr according to
            # multiCycleLR schedule.
            self.assertLessEqual(np.max(lr), current_lr_conf["max_lr"])
            # Check that lr of this epoch stays above min lr.
            self.assertGreaterEqual(
                np.round(np.min(lr), 8),
                np.round(initial_lr / current_lr_conf["final_div_factor"], 8),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
