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

import ray
import torch

from nupic.research.frameworks.dynamic_sparse.common.loggers import DEFAULT_LOGGERS
from nupic.research.frameworks.dynamic_sparse.common.utils import (
    Trainable,
    new_experiment,
    run_experiment,
)
from nupic.research.frameworks.pytorch.model_utils import set_random_seed

# Set seed for `random`, `numpy`, and `pytorch`.
set_random_seed(32)


def serializer(obj):
    if obj.is_cuda:
        return obj.cpu().numpy()
    else:
        return obj.numpy()


def deserializer(serialized_obj):
    return serialized_obj


# experiment configurations
base_exp_config = dict(
    device=("cuda" if torch.cuda.device_count() > 0 else "cpu"),
    # dataset related
    dataset_name="PreprocessedGSC",
    data_dir="~/nta/datasets/gsc",
    batch_size_train=16,
    batch_size_test=(1000),
    # network related
    network="GSCHeb",
    # ----- Optimizer Related ----
    optim_alg="SGD",
    momentum=0.0,
    learning_rate=0.01,
    weight_decay=1e-2,
    # ----- LR Scheduler Related ----
    lr_scheduler="StepLR",
    lr_step_size=1,
    lr_gamma=0.9,
    # additional validation
    test_noise=False,
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
experiment_name = "gsc-trials-2019-10-03-test"
tune_config = dict(
    name=experiment_name,
    num_samples=5,
    local_dir=os.path.expanduser(os.path.join("~/nta/results", experiment_name)),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 30},
    resources_per_trial={
        "cpu": os.cpu_count() / 2.0,
        "gpu": torch.cuda.device_count() / 2.0,
    },
    loggers=DEFAULT_LOGGERS,
    verbose=1,
    config=base_exp_config,
)

# define experiments
experiments = {
    "gsc-BaseModel": dict(model=ray.tune.grid_search(["BaseModel"])),
    "gsc-Static": dict(
        model=ray.tune.grid_search(["SparseModel"]),
        # sparse related
        on_perc=ray.tune.grid_search(
            [
                [None, None, 0.4, None],
                [None, None, 0.1, None],
                [None, None, 0.05, None],
                [None, None, 0.025, None],
            ]
        ),
    ),
    "gsc-Heb": dict(
        model=ray.tune.grid_search(["DSNNMixedHeb"]),
        # network related
        prune_methods=[None, None, "dynamic-linear", None],
        # sparse related
        on_perc=ray.tune.grid_search(
            [
                [None, None, 0.4, None],
                [None, None, 0.1, None],
                [None, None, 0.05, None],
                [None, None, 0.025, None],
            ]
        ),
        hebbian_prune_perc=0.3,
        hebbian_grow=True,
        weight_prune_perc=None,
        moving_average_alpha=ray.tune.grid_search([0.6]),
        use_binary_coactivations=False,
        # debug related
        log_magnitude_vs_coactivations=True,
    ),
    "gsc-SET": dict(
        model=ray.tune.grid_search(["DSNNMixedHeb"]),
        # network related
        prune_methods=[None, None, "dynamic-linear", None],
        # sparse related
        on_perc=ray.tune.grid_search(
            [
                [None, None, 0.4, None],
                [None, None, 0.1, None],
                [None, None, 0.05, None],
                [None, None, 0.025, None],
            ]
        ),
        hebbian_prune_perc=None,
        hebbian_grow=False,
        weight_prune_perc=0.3,
        # debugging related
        log_magnitude_vs_coactivations=True,
    ),
}
exp_configs = (
    [(name, new_experiment(base_exp_config, c)) for name, c in experiments.items()]
    if experiments
    else [(experiment_name, base_exp_config)]
)

# Register serializers.
ray.init()
for t in [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.HalfTensor,
    torch.ByteTensor,
    torch.CharTensor,
    torch.ShortTensor,
    torch.IntTensor,
    torch.LongTensor,
    torch.Tensor,
]:
    ray.register_custom_serializer(t, serializer=serializer, deserializer=deserializer)

# run all experiments in parallel
results = [
    run_experiment.remote(name, Trainable, c, tune_config) for name, c in exp_configs
]
ray.get(results)
ray.shutdown()
