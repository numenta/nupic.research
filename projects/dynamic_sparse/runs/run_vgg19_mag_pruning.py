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

from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)
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
    # ----- Dataset Related -----
    # > CIFAR10
    dataset_name="CIFAR10",
    data_dir="~/nta/datasets",
    # ----- Network Related ------
    # > Dynamic-VGG19
    network="vgg19_dsnn",
    init_weights=True,
    kwinners=True,
    boost_strength=1.5,
    boost_strength_factor=0.85,
    percent_on=0.3,
    # ----- Optimizer Related ----
    # > GSCSparseCNN
    optim_alg="SGD",
    momentum=0.9,
    learning_rate=0.1,
    weight_decay=0.0005,
    # ----- LR Scheduler Related ----
    lr_scheduler="MultiStepLR",
    lr_milestones=[81, 122],
    lr_gamma=0.1,
    # ----- Batch-size info ----
    train_batch_size=(4, 128),
    train_batches_per_epoch=(600, 400),
    test_batch_size=128,
    test_batches_per_epoch=500,
    # ----- Dynamic-Sparse Related -----
    # * See models for allowable params
    # * Values entered below in experiments.
    # ----- Debugging -----
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
experiment_name = __file__.replace(".py", "")
tune_config = dict(
    name=experiment_name + "_eval-0",
    num_samples=1,
    local_dir=os.path.expanduser(os.path.join("~/nta/results", experiment_name)),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 200},
    resources_per_trial={
        "cpu": os.cpu_count() / 2,
        "gpu": torch.cuda.device_count() / 2,
    },
    loggers=DEFAULT_LOGGERS,
    verbose=1,
    config=base_exp_config,
)

# define experiments
experiments = {
    "dense-baseline": dict(
        model=ray.tune.grid_search(["BaseModel"]),
        prune_methods=[None] * 17,
        update_nsteps=[None] * 17,
        on_perc=[1.0] * 17,
    ),
    "model-comparison": dict(
        model=ray.tune.grid_search(["DSNNWeightedMag", "DSNNMixedHeb", "SparseModel"]),
        prune_methods=["dynamic-conv"] + [None] * 16,
        update_nsteps=[1] + [None] * 16,
        on_perc=[0.50] * 16 + [1.0],
        hebbian_prune_perc=None,
        hebbian_grow=False,
        weight_prune_perc=0.3,
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
