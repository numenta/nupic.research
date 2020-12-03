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

import torch

from nupic.research.frameworks.lightning.models import SupervisedModel
from nupic.research.frameworks.pytorch.datasets import torchvisiondataset
from nupic.research.frameworks.pytorch.models import StandardMLP

"""
Base Experiment to test MNIST
"""

EPOCHS = 90
base_mnist = dict(
    lightning_model_class=SupervisedModel,
    lightning_model_args=dict(
        config=dict(
            dataset_class=torchvisiondataset,
            model_class=StandardMLP,
            model_args=dict(
                input_size=(28, 28),
                num_classes=10,
            ),
            dataset_args=dict(
                root="~/nta/datasets",
                dataset_name="MNIST",
                download=True,
            ),

            batch_size=128,

            epochs=EPOCHS,

            optimizer_class=torch.optim.SGD,
            optimizer_args=dict(
                lr=0.1,
                weight_decay=1e-6,
                momentum=0.9,
                nesterov=False,
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args=dict(
                gamma=0.1,
                step_size=30,
            ),
        ),
    ),

    lightning_trainer_args=dict(
        # No early stopping
        min_epochs=EPOCHS,
        max_epochs=EPOCHS,
    )
)

CONFIGS = dict(
    base_mnist=base_mnist,
)
