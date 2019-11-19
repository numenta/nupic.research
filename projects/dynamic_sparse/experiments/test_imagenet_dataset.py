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


from ray import tune
from nupic.research.frameworks.dynamic_sparse.common.utils import run_ray

# experiment configurations
base_exp_config = dict(
    device="cuda",
    dataset_name="ImageNet",
    use_multiple_gpus=True,
    model="PruningModel",
    data_dir="~/nta/datasets",
    num_classes=1000,
    epochs=20,
    batch_size_train=1024,
    batch_size_test=1024, # 5120,
    # ---- network related
    network="resnet50",
    pretrained=True,
    # ---- optimizer related
    optim_alg="Adam",
    learning_rate=1e-4,
    weight_decay=1e-2,
    # ---- sparsity related
    on_perc=1.0,
    target_final_density=0.2,
    sparse_start=None,
    sparse_end=None,
)

# ray configurations
tune_config = dict(
    num_samples=1,
    name=__file__.replace(".py", ""),
    checkpoint_freq=0,
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 60, "gpu": 8},
    verbose=2,
)

run_ray(tune_config, base_exp_config)
