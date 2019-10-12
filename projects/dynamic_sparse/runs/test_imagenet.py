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
    dataset_name="CIFAR10",
    model="BaseModel",
    data_dir="~/nta/datasets",
    epochs=120,
    # ---- network related
    #network="resnet152",
    # network="WideResNet",
    network=tune.grid_search(["resnet152", "WideResNet"]),
    percent_on_k_winner=tune.grid_search([0.25, 1]),
    boost_strength=1.4,
    boost_strength_factor=0.7,
    k_inference_factor=1.0,            
    # wideresnet parameters
    # widen_factor=8,
    # depth=28,
    # dropout_rate=0.3,
    # ---- optimizer related
    optim_alg="SGD",
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    lr_milestones=[60, 90, 110],
    # lr_milestones=[60, 120, 160],
    lr_gamma=0.2,
    weight_decay=0.0005,
    momentum=0.9,
    # ---- debugs and noise related    
    test_noise=True,
    noise_level=0.15,
)

# ray configurations
tune_config = dict(
    num_samples=1,
    name=__file__.replace(".py", "") + "4",
    checkpoint_freq=0,
    checkpoint_at_end=False,
    resources_per_trial={"cpu": 1, "gpu": .5},
    verbose=0,
)

run_ray(tune_config, base_exp_config)
