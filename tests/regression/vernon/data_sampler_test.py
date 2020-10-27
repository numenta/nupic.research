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
from copy import deepcopy

import torch

from nupic.research.frameworks.pytorch.datasets import omniglot
from nupic.research.frameworks.pytorch.models import OMLNetwork
from nupic.research.frameworks.vernon import MetaContinualLearningExperiment

ROOT = "/mnt/datasets"

CONFIG = dict(
    # training infrastructure
    distributed=True,

    # problem specific
    experiment_class=MetaContinualLearningExperiment,
    dataset_class=omniglot,
    dataset_args=dict(
        root=ROOT,
        download=False,  # This should already be downloaded,
    ),
    model_class=OMLNetwork,
    model_args=dict(num_classes=963),

    # optimizer
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=1e-4),

    # metacl variables
    num_classes=963,
    num_classes_eval=660,
    batch_size=1,
    val_batch_size=15,
    slow_batch_size=5,
    replay_batch_size=15,
    num_fast_steps=5,  # => we take 5 * 'tasks_per_epoch' sequential grad steps
    train_train_sample_size=20,  # This applies to both the slow and fast iterations.
    epochs=10,
    tasks_per_epoch=3,
    run_meta_test=False,  # we won't run this for now
)


class DataSamplerTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_meta_cl_with_distributed_sampler(self):
        """Start a meta-cl experiment and validate the distributed sampler."""

        exp = CONFIG["experiment_class"]()
        exp.setup_experiment(CONFIG)

        # Test 20 images in train_train sampler
        train_fast_loader = exp.train_fast_loader
        train_fast_sampler = train_fast_loader.sampler
        train_fast_sampler.set_active_tasks([0, 1, 2])

        # Fast loader should have 20 images for each of the three tasks.
        self.assertTrue(len(train_fast_loader), 60)  # == 3 * 20

        # Test 20 images in train_test sampler
        train_slow_loader = exp.train_slow_loader
        train_slow_sampler = train_slow_loader.sampler
        train_slow_sampler.set_active_tasks([0, 1, 2])

        # Slow loader should have 60 images with a batch size of 5
        self.assertTrue(len(train_slow_loader), 12)  # == 60 / 5

        # Validate classes in tasks being sampled.
        fast_tasks = []
        for _, y in train_fast_loader:
            fast_tasks.append(y.item())
        self.assertTrue(set(fast_tasks), set([0, 1, 2]))

    def test_meta_cl_with_non_distributed_sampler(self):
        """Start a meta-cl experiment and validate the non-distributed sampler."""

        # Test non-distributed sampler.
        config_nondist = deepcopy(CONFIG)
        config_nondist.update(distributed=False)

        exp_nondist = config_nondist["experiment_class"]()
        exp_nondist.setup_experiment(config_nondist)

        # Test 20 images in train_train sampler
        train_fast_loader = exp_nondist.train_fast_loader
        train_fast_sampler = train_fast_loader.sampler
        train_fast_sampler.set_active_tasks([0, 1, 2])

        # Fast loader should have 20 images for each of the three tasks.
        self.assertEqual(len(train_fast_loader), 60)  # == 3 * 20

        # Test 20 images in train_test sampler
        train_slow_loader = exp_nondist.train_slow_loader
        train_slow_sampler = train_slow_loader.sampler
        train_slow_sampler.set_active_tasks([0, 1, 2])

        # Slow loader should have 60 images with a batch size of 5
        self.assertEqual(len(train_slow_loader), 12)  # == 60 / 5

        # Validate classes in tasks being sampled.
        fast_tasks = []
        for _, y in train_fast_loader:
            fast_tasks.append(y.item())
        self.assertEqual(set(fast_tasks), set([0, 1, 2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
