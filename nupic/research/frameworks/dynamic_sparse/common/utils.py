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
from copy import deepcopy

import ray
import torch  # to remove later
from ray import tune
from torchvision import datasets

import nupic.research.frameworks.dynamic_sparse.models as models
import nupic.research.frameworks.dynamic_sparse.networks as networks
from nupic.research.frameworks.pytorch.model_utils import set_random_seed

from .datasets import Dataset

class Trainable(tune.Trainable):
    """ray.tune trainable generic class Adaptable to any pytorch module."""

    def __init__(self, config=None, logger_creator=None):
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        network = getattr(networks, config["network"])(config=config)
        # summary(network.to(torch.device('cpu')), input_size=(1, 32, 32))
        self.model = getattr(models, config["model"])(network, config=config)
        self.dataset = Dataset(config=config)
        self.model.setup()

    def _train(self):
        log = self.model.run_epoch(self.dataset, self._iteration)
        return log

    def _save(self, checkpoint_dir):
        self.model.save(checkpoint_dir)

    def _restore(self, checkpoint):
        self.model.restore(checkpoint)


def download_dataset(config):
    """Pre-downloads dataset.
    Required to avoid multiple simultaneous attempts to download same
    dataset
    """
    dataset_name = config["dataset_name"]
    if hasattr(datasets, dataset_name):
        getattr(datasets, config["dataset_name"])(
            download=True, root=os.path.expanduser(config["data_dir"])
        )


def new_experiment(base_config, new_config):
    modified_config = deepcopy(base_config)
    modified_config.update(new_config)
    return modified_config


@ray.remote
def run_experiment(name, trainable, exp_config, tune_config):

    # override when running local for test
    # if not torch.cuda.is_available():
    #     exp_config["device"] = "cpu"
    #     tune_config["resources_per_trial"] = {"cpu": 1}

    # download dataset
    download_dataset(exp_config)

    # run
    tune_config["name"] = name
    tune_config["config"] = exp_config
    tune.run(Trainable, **tune_config)


def init_ray():

    ray.init()

    def serializer(obj):
        if obj.is_cuda:
            return obj.cpu().numpy()
        else:
            return obj.numpy()

    def deserializer(serialized_obj):
        return serialized_obj

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
        ray.register_custom_serializer(
            t, serializer=serializer, deserializer=deserializer
        )


def run_ray(tune_config, exp_config, fix_seed=False):

    # update config
    tune_config["config"] = exp_config
    download_dataset(exp_config)

    # override when running local for test
    if not torch.cuda.is_available():
        tune_config["config"]["device"] = "cpu"
        tune_config["resources_per_trial"] = {"cpu": 1}

    # move epochs to tune_config, to keep track
    if "stop" not in tune_config:
        if "epochs" in exp_config:
            tune_config["stop"] = {"training_iteration": exp_config["epochs"]}

    # expand path in dir
    if "local_dir" in tune_config:
        tune_config["local_dir"] = os.path.expanduser(tune_config["local_dir"])
    else:
        tune_config["local_dir"] = os.path.expanduser("~/nta/results")

    # init ray
    ray.init(load_code_from_local=True)

    # MC code to fix for an unknown bug
    def serializer(obj):
        if obj.is_cuda:
            return obj.cpu().numpy()
        else:
            return obj.numpy()

    def deserializer(serialized_obj):
        return serialized_obj

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
        ray.register_custom_serializer(
            t, serializer=serializer, deserializer=deserializer
        )

    # fix seed
    if fix_seed:
        set_random_seed(32)

    tune.run(Trainable, **tune_config)


def run_ray_many(tune_config, exp_config, experiments, fix_seed=False):

    # update config
    tune_config["config"] = exp_config

    # override when running local for test
    if not torch.cuda.is_available():
        tune_config["config"]["device"] = "cpu"
        tune_config["resources_per_trial"] = {"cpu": 1}

    # MC code to fix for an unknown bug
    def serializer(obj):
        if obj.is_cuda:
            return obj.cpu().numpy()
        else:
            return obj.numpy()

    def deserializer(serialized_obj):
        return serialized_obj

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
        ray.register_custom_serializer(
            t, serializer=serializer, deserializer=deserializer
        )

    # fix seed
    if fix_seed:
        set_random_seed(32)

    # multiple experiments
    exp_configs = [
        (name, new_experiment(exp_config, c)) for name, c in experiments.items()
    ]

    # init ray
    ray.init()
    results = [
        run_experiment.remote(name, Trainable, c, tune_config)
        for name, c in exp_configs
    ]
    ray.get(results)
    ray.shutdown()
