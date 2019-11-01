# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import os

import numpy as np

from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

exp_config = dict(
    device="cuda",
    network="MLP",
    dataset_name="MNIST",
    input_size=784,
    hidden_sizes=[50, 50, 50],
    model="IterativePruningModel",
    epochs=2,
    train_batches_per_epoch=2,
    # ---- sparsity related
    experiment_type="IterativePruning",
    # 0.2, 0.4, 0.6, 0.8, 1.0
    iterative_pruning_schedule=list(np.arange(0.2, 1.01, 0.20)),
    sparse_start=None,
    sparse_end=None,
    on_perc=1.0,
    # ---- optimizer related
    optim_alg="SGD",
    learning_rate=0.1,
    weight_decay=0,
)

# run
tune_config = dict(
    name=__file__.replace(".py", "") + "_lt",
    num_samples=1,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    resources_per_trial={"cpu": 1, "gpu": 1},
    verbose=0,
)

run_ray(tune_config, exp_config, fix_seed=True)

# 10/31 - ran script, working ok, results as expected
