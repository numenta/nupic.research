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
    download_dataset,
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
    # dataset_name="CIFAR10",
    # input_size=(3, 32, 32),
    # > MNIST
    # dataset_name="MNIST",
    # input_size=(1, 28, 28),
    # data_dir="~/nta/datasets/gsc",
    # stats_mean=(0.4914, 0.4822, 0.4465),
    # stats_std=(0.2023, 0.1994, 0.2010),
    # > GSC
    dataset_name="PreprocessedGSC",
    data_dir="~/nta/datasets/gsc",
    batch_size_train=(4, 16),
    batch_size_test=(1000),
    # ----- Network Related ------
    # > DSCNN
    # model="DSCNN",
    # network="mnist_sparse_dscnn",
    # > GSCSparseCNN
    model="BaseModel",
    network="gsc_sparse_cnn",
    # init_weights=True,
    # batch_norm=True,
    # dropout=False,
    # kwinners=True,
    # percent_on=0.3,
    # boost_strength=1.4,
    # boost_strength_factor=0.7,
    # ----- Optimizer Related ----
    # optim_alg="SGD",
    # momentum=0.9,
    # learning_rate=0.01,
    # weight_decay=1e-4,
    # > GSCSparseCNN
    optim_alg="SGD",
    momentum=0.0,
    learning_rate=0.01,
    weight_decay=1e-2,
    # ----- LR Scheduler Related ----
    lr_scheduler="StepLR",
    lr_step_size=1,
    lr_gamma=0.9,
    # ----- Dynamic-Sparse Related -----
    # * See DSCNN network for allowable params
    # * Params set below in experiments.
    # ----- Additional Validation -----
    test_noise=False,
    noise_level=0.1,
    # ----- Vebugging -----
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
experiment_name = "gsc-sparse-cnn-pruning-comparisons-0-2019-08-22"
tune_config = dict(
    name=experiment_name,
    num_samples=1,
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
    # "gsc-baseline": dict(
    #     model="DSCNN",
    #     network="gsc_sparse_dscnn",
    #     prune_methods=["none", "none"],
    # ),
    # "dynamic-hebbian-second-layer-99_9-sparse": dict(
    #     model="DSCNN",
    #     network="gsc_sparse_dscnn",
    #     prune_methods=["none", "dynamic"],
    #     hebbian_prune_frac=0.9995,
    #     magnitude_prune_frac=0.0,
    #     sparsity=0.999,
    #     prune_dims=tuple(),
    # ),
    # "dynamic-hebbian-second-layer-99-sparse": dict(
    #     model="DSCNN",
    #     network="gsc_sparse_dscnn",
    #     prune_methods=["none", "dynamic"],
    #     hebbian_prune_frac=0.995,
    #     magnitude_prune_frac=0.0,
    #     sparsity=0.99,
    #     prune_dims=tuple(),
    # ),
    "dynamic-hebbian-second-layer-98-sparse": dict(
        model="DSCNN",
        network="gsc_sparse_dscnn",
        prune_methods=["none", "dynamic"],
        hebbian_prune_frac=0.99,
        magnitude_prune_frac=0.0,
        sparsity=0.98,
        update_nsteps=50,
        prune_dims=tuple(),
    ),
    # "static-second-layer-varying-sparsity": dict(
    #     model="DSCNN",
    #     network="gsc_sparse_dscnn",
    #     prune_methods=["none", "static"],
    #     sparsity=tune.grid_search([0.98, 0.99, 0.999]),
    # ),
}
exp_configs = (
    [(name, new_experiment(base_exp_config, c)) for name, c in experiments.items()]
    if experiments
    else [(experiment_name, base_exp_config)]
)

# Download dataset.
download_dataset(base_exp_config)

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
