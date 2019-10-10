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
    network="WideResNet",
    model="BaseModel",
    epochs=200,
    # ---- network related
    num_classes=200,
    dropout_rate=0.3,
    # ---- optimizer related
    optim_alg="SGD",
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    lr_milestones=[60, 120, 160],
    lr_gamma=0.2,
    weight_decay=0.0005,
    momentum=0.9,
    widen_factor=8,
    depth=28
)

# ray configurations
tune_config = dict(
    num_samples=1,
    name=__file__.replace(".py", ""),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    resources_per_trial={"cpu": 1, "gpu": 1},
    verbose=0,
)

run_ray(tune_config, base_exp_config)
