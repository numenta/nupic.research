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

from copy import deepcopy

from nupic.research.frameworks.greedy_infomax.models import SparseFullVisionModel

from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS

BATCH_SIZE = 32
NUM_EPOCHS = 60
DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]
SPARSE_BASE = deepcopy(DEFAULT_BASE)
SPARSE_BASE.update(
    dict(
        wandb_args=dict(project="greedy_infomax-sparsity", name="sparse_resnet_base"),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        model_class=SparseFullVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
            sparsity=[0.5, 0.5, 0.5],
        ),
    )
)

CONFIGS = dict(sparse_base=SPARSE_BASE)
