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
    dataset_name="MNIST",
    data_dir=os.path.expanduser("~/nta/datasets"),
    input_size=784,
    num_classes=10,
    # network related
    network="MLPHeb",
    hidden_sizes=[100, 100, 100],
    batch_norm=True,
    # model related
    optim_alg="SGD",
    momentum=0.9,
    weight_decay=1e-4,
    learning_rate=0.1,
    lr_scheduler="MultiStepLR",
    lr_milestones=[30, 60, 90],
    lr_gamma=0.1,
    # additional validation
    test_noise=False,
    # debugging
    debug_weights=True,
    debug_sparse=True,
)

# ray configurations
experiment_name = "mlp-heb-coacts-comparison-2019-10-01-C"
tune_config = dict(
    name=experiment_name,
    num_samples=1,
    local_dir=os.path.expanduser(os.path.join("~/nta/results", experiment_name)),
    checkpoint_freq=0,
    checkpoint_at_end=False,
    stop={"training_iteration": 5},
    resources_per_trial={
        "cpu": os.cpu_count() / 4.0,
        "gpu": torch.cuda.device_count() / 4.0,
    },
    loggers=DEFAULT_LOGGERS,
    verbose=1,
    config=base_exp_config,
)

# define experiments
experiments = {
    # "mlp-dense-relu": dict(
    #     model=ray.tune.grid_search(["BaseModel"]),
    # ),
    # "mlp-sparse-relu": dict(
    #     model=ray.tune.grid_search(["SparseModel"]),
    #     on_perc=ray.tune.grid_search([0.02]),
    # ),
    # "mlp-SET-relu": dict(
    #     model=ray.tune.grid_search(["DSNNMixedHeb"]),
    #     on_perc=ray.tune.grid_search([0.02]),
    #     hebbian_prune_perc=None,
    #     hebbian_grow=False,
    #     weight_prune_perc=0.3,
    #     pruning_early_stop=2,
    # ),
    "mlp-Heb-coacts-xixj-relu-plots": dict(
        model=ray.tune.grid_search(["DSNNMixedHeb"]),
        on_perc=ray.tune.grid_search([0.02]),
        log_magnitude_vs_coactivations=True,
        hebbian_prune_perc=0.3,
        hebbian_grow=ray.tune.grid_search([True]),
        weight_prune_perc=None,
        pruning_early_stop=2,
        moving_average_alpha=ray.tune.grid_search([0.6]),
        reset_coactivations=ray.tune.grid_search([True]),
        use_binary_coactivations=False,
    ),
    # "mlp-hebbian-1": dict(
    #     model=ray.tune.grid_search(["DSNNMixedHeb"]),
    #     # sparse related
    #     on_perc=0.1,
    #     hebbian_prune_perc=0.3,
    #     hebbian_grow=False,
    #     weight_prune_perc=None,
    #     pruning_early_stop=2,
    #     log_magnitude_vs_coactivations=True,
    # ),
    # "gsc-hebbian-0": dict(
    #     model=ray.tune.grid_search(["DSNNMixedHeb"]),
    #     # network related
    #     network="gsc_sparse_dsnn",
    #     prune_methods=[None, "dynamic-conv", "dynamic-linear", None],
    #     # dataset related
    #     dataset_name="PreprocessedGSC",
    #     data_dir="~/nta/datasets/gsc",
    #     batch_size_train=(4, 16),
    #     batch_size_test=(1000),
    #     # sparse related
    #     on_perc=0.1,
    #     hebbian_prune_perc=0.3,
    #     hebbian_grow=False,
    #     weight_prune_perc=None,
    #     pruning_early_stop=2,
    #     log_magnitude_vs_coactivations=True,
    # ),
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
