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
    dataset_name="TinyImageNet",
    model=tune.grid_search(["BaseModel", "SparseModel"]),
    data_dir="~/nta/datasets",
    num_classes=200,
    epochs=200,
    # epochs = 120,
    # ---- network related
    network=tune.grid_search(["resnet50", "resnet50_pretrained"]),
    percent_on_k_winner=tune.grid_search([0.25, 1]),
    boost_strength=1.5,
    boost_strength_factor=0.85,
    k_inference_factor=1.0,
    # ---- sparse model related
    sparse_type="precise_per_output",
    sparse_start=1,
    sparse_end=None,
    on_perc=0.5,
    # ---- optimizer related
    optim_alg="SGD",
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    # lr_milestones=[60, 90, 110],
    lr_milestones=[60, 120, 160],
    lr_gamma=0.2,
    weight_decay=0.0005,
    momentum=0.9,
    nesterov_momentum=True,
    # ---- debugs and noise related
    test_noise=True,
    noise_level=0.15,
)

# ray configurations
tune_config = dict(
    num_samples=1,
    name=__file__.replace(".py", "") + "_test",
    checkpoint_freq=0,
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 1, "gpu": 0.5},
    verbose=0,
)

run_ray(tune_config, base_exp_config)
