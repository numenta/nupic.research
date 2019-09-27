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

from ray.tune.logger import DEFAULT_LOGGERS, MLFLowLogger

from mlflow.tracking import MlflowClient
from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# alternative initialization based on configuration
exp_config = dict(
    device="cuda",
    network="MLPHeb",
    dataset_name="MNIST",
    input_size=784,
    num_classes=10,
    model="SparseModel",
    data_dir="~/nta/data",
    on_perc=0.2,
    batch_size_train=10,
    batch_size_test=10,
    debug_sparse=True,
)

client = MlflowClient()
exp_config["mlflow_experiment_id"] = client.create_experiment("test_mlflow5")

# run
tune_config = dict(
    name=__file__,
    num_samples=3,
    local_dir=os.path.expanduser("~/nta/results"),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 5},
    resources_per_trial={"cpu": 1, "gpu": 1},
    verbose=2,
    loggers=DEFAULT_LOGGERS + (MLFLowLogger,),
)

run_ray(tune_config, exp_config)

# df = mlflow.search_runs([experiment_id])
