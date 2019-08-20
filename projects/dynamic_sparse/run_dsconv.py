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
import ray.tune as tune
import torch

from loggers import DEFAULT_LOGGERS
from utils import Trainable, download_dataset, new_experiment, run_experiment

torch.manual_seed(32)


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
    # dataset_name="CIFAR10",
    # input_size=(3, 32, 32),
    dataset_name="MNIST",
    input_size=(1, 28, 28),
    num_classes=10,
    # stats_mean=(0.4914, 0.4822, 0.4465),
    # stats_std=(0.2023, 0.1994, 0.2010),
    data_dir="~/nta/datasets",

    # ----- Network Related ------
    model="DSCNN",
    network="mnist_sparse_dscnn",
    # init_weights=True,
    # batch_norm=True,
    # dropout=False,
    # kwinners=True,
    # percent_on=0.3,
    # boost_strength=1.4,
    # boost_strength_factor=0.7,

    # ----- Optimizer Related ----
    optim_alg="SGD",
    momentum=0.9,
    learning_rate=0.01,
    weight_decay=1e-4,

    # ----- Dynamic-Sparse Related -----
    #  todo ...

    # ----- Additional Validation -----
    test_noise=False,
    noise_level=0.1,

    # ----- Vebugging -----
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
experiment_name = "dscnn-dynamic-by-weight-2019-08-19"
tune_config = dict(
    name=experiment_name,
    num_samples=7,
    local_dir=os.path.expanduser(os.path.join("~/nta/results", experiment_name)),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 15},
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
    # "normal-baseline": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'none'],
    # ),
    # "static-first-layer-baseline-80-sparse": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['static', 'none'],
    #     sparsity=0.80,
    # ),
    # "static-second-layer-baseline-98-sparse": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'static'],
    #     sparsity=0.98,
    # ),
    # "static-both-layers-baseline-80-98-sparse": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['static', 'static'],
    #     sparsity=[0.80, 0.98],
    # ),

    # "random-baseline": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'random'],
    #     hebbian_prune_frac=0.90,
    #     weight_prune_frac=0.0,
    #     sparsity=tune.grid_search([0.80, 0.90, 0.98]),
    #     prune_dims=[[], []],
    # ),

    "dynamic-first-layer-80-sparse-by-weight": dict(
        network="mnist_sparse_dscnn",
        prune_methods=['dynamic', 'none'],
        hebbian_prune_frac=0.0,
        weight_prune_frac=0.90,
        sparsity=0.80,
        prune_dims=[[], []],
    ),
    "dynamic-second-layer-98-sparse-by-weight": dict(
        network="mnist_sparse_dscnn",
        prune_methods=['none', 'dynamic'],
        hebbian_prune_frac=0.0,
        weight_prune_frac=0.99,
        sparsity=0.98,
        prune_dims=[[], []],
    ),

    # "dynamic-first-layer-80-sparse": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_method=['dynamic', 'none'],
    #     hebbian_prune_frac=0.9,
    #     weight_prune_frac=0.0,
    #     sparsity=0.8,
    #     prune_dims=[[], []],
    # ),

    "dynamic-both-layers-80-98-sparse-by-weight": dict(
        network="mnist_sparse_dscnn",
        prune_method=['dynamic', 'dynamic'],
        hebbian_prune_frac=0.0,
        weight_prune_frac=[0.90, 0.99],
        sparsity=[0.8, 0.98],
        prune_dims=[[], []],
    ),

    # "dynamic-one-layer-0a": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.80,
    #     weight_prune_frac=0.0,
    #     sparsity=0.79,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-1a": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.90,
    #     weight_prune_frac=0.0,
    #     sparsity=0.89,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-2a": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_method=['none', 'dynamic'],
    #     hebbian_prune_frac=0.96,
    #     weight_prune_frac=0.0,
    #     sparsity=0.95,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-3a": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.99,
    #     weight_prune_frac=0.0,
    #     sparsity=0.98,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-4a-varying-sparsity": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.90,
    #     weight_prune_frac=0.0,
    #     sparsity=tune.grid_search([0.7, 0.8, 0.85, 0.88, 0.89]),
    #     prune_dims=[[], []],
    # ),

    # "dynamic-one-layer-0b": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['dynamic', 'none'],
    #     hebbian_prune_frac=0.5,
    #     weight_prune_frac=0.0,
    #     sparsity=0.4,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-1b": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['dynamic', 'none'],
    #     hebbian_prune_frac=0.7,
    #     weight_prune_frac=0.0,
    #     sparsity=0.6,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-2b": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['dynamic', 'none'],
    #     hebbian_prune_frac=0.8,
    #     weight_prune_frac=0.0,
    #     sparsity=0.6,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-3b": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_method=['dynamic', 'none'],
    #     hebbian_prune_frac=0.9,
    #     weight_prune_frac=0.0,
    #     sparsity=0.8,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-4b-varying-sparsity": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.7,
    #     weight_prune_frac=0.0,
    #     sparsity=tune.grid_search([0.5, 0.60, 0.65, 0.68]),
    #     prune_dims=[[], []],
    # ),

    # "dynamic-two-layer-0": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['dynamic', 'dynamic'],
    #     hebbian_prune_frac=[0.80, 0.9],
    #     weight_prune_frac=0.0,
    #     sparsity=[0.75, 0.88],
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-1": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['dynamic', 'dynamic'],
    #     hebbian_prune_frac=[0.85, 0.95],
    #     weight_prune_frac=0.0,
    #     sparsity=[0.8, 0.93],
    #     prune_dims=[[], []],
    # ),

    # "dynamic-one-layer-weight-v-heb-1": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.9,
    #     weight_prune_frac=tune.grid_search([0.5, 0.7, 0.9, 0.96]),
    #     sparsity=0.8,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-weight-v-heb-2": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.96,
    #     weight_prune_frac=tune.grid_search([0.5, 0.7, 0.9, 0.96]),
    #     sparsity=0.9,
    #     prune_dims=[[], []],
    # ),
    # "dynamic-one-layer-weight-v-heb-3": dict(
    #     network="mnist_sparse_dscnn",
    #     prune_methods=['none', 'dynamic'],
    #     hebbian_prune_frac=0.99,
    #     weight_prune_frac=tune.grid_search([0.5, 0.7, 0.9, 0.99]),
    #     sparsity=0.98,
    #     prune_dims=[[], []],
    # ),

}
exp_configs = [
    (name, new_experiment(base_exp_config, c)) for name, c in experiments.items()
]

# Download dataset.
download_dataset(base_exp_config)

# Register serializers.
ray.init()
for t in [
    torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor,
    torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
    torch.IntTensor, torch.LongTensor, torch.Tensor
]:
    ray.register_custom_serializer(
        t, serializer=serializer, deserializer=deserializer)

# run all experiments in parallel
results = [
    run_experiment.remote(name, Trainable, c, tune_config) for name, c in exp_configs
]
ray.get(results)
ray.shutdown()
