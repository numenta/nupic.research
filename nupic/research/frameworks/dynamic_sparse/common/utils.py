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
from collections.abc import Iterable
from copy import deepcopy

import ray
import torch  # to remove later
from ray import tune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nupic.research.frameworks.dynamic_sparse.models as models
import nupic.research.frameworks.dynamic_sparse.networks as networks
from nupic.research.frameworks.pytorch.image_transforms import RandomNoise
from nupic.research.frameworks.pytorch.model_utils import set_random_seed

from nupic.research.frameworks.dynamic_sparse.common.dataloaders import PreprocessedSpeechDataLoader, VaryingDataLoader


class Dataset:
    """Loads a dataset.
    Returns object with a pytorch train and test loader
    """

    def __init__(self, config=None):

        defaults = dict(
            dataset_name=None,
            data_dir=None,
            batch_size_train=128,
            batch_size_test=128,
            stats_mean=None,
            stats_std=None,
            augment_images=False,
            test_noise=False,
            noise_level=0.1,
        )
        defaults.update(config)
        self.__dict__.update(defaults)
        self.data_dir = os.path.expanduser(self.data_dir)

        if hasattr(datasets, self.dataset_name):
            self.load_from_torch_vision()
        elif self.dataset_name == "PreprocessedGSC":
            self.load_preprocessed_gsc()
        else:
            raise Exception("Dataset {}")

    def load_preprocessed_gsc(self):

        self.train_loader = PreprocessedSpeechDataLoader(
            self.data_dir,
            subset="train",
            batch_sizes=self.batch_size_train,
            shuffle=True,
        )

        self.test_loader = PreprocessedSpeechDataLoader(
            self.data_dir,
            subset="valid",
            silence_percentage=0,
            batch_sizes=self.batch_size_test,
        )

        if self.test_noise:
            self.noise_loader = PreprocessedSpeechDataLoader(
                self.data_dir,
                subset="test_noise",
                silence_percentage=0,
                batch_sizes=self.batch_size_test,
            )
        else:
            self.noise_loader = None

    def load_from_torch_vision(self):

        # special dataloader case
        if isinstance(self.batch_size_train, Iterable) or isinstance(
            self.batch_size_test, Iterable
        ):
            dataloader_type = VaryingDataLoader
        else:
            dataloader_type = DataLoader

        # expand ~
        self.data_dir = os.path.expanduser(self.data_dir)

        # recover mean and std to normalize dataset
        if not self.stats_mean or not self.stats_std:
            tempset = getattr(datasets, self.dataset_name)(
                root=self.data_dir, train=True, transform=transforms.ToTensor()
            )
            self.stats_mean = (tempset.data.float().mean().item() / 255,)
            self.stats_std = (tempset.data.float().std().item() / 255,)
            del tempset

        # set up transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.stats_mean, self.stats_std),
            ]
        )
        # set up augment transforms for training
        if not self.augment_images:
            aug_transform = transform
        else:
            aug_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.stats_mean, self.stats_std),
                ]
            )

        # load train set
        train_set = getattr(datasets, self.dataset_name)(
            root=self.data_dir, train=True, transform=aug_transform
        )
        self.train_loader = dataloader_type(
            dataset=train_set, batch_size=self.batch_size_train, shuffle=True
        )

        # load test set
        test_set = getattr(datasets, self.dataset_name)(
            root=self.data_dir, train=False, transform=transform
        )
        self.test_loader = dataloader_type(
            dataset=test_set, batch_size=self.batch_size_test, shuffle=False
        )

        # noise dataset
        if self.test_noise:
            noise = self.noise_level
            noise_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(self.stats_mean, self.stats_std),
                    RandomNoise(
                        noise, high_value=0.5 + 2 * 0.20, low_value=0.5 - 2 * 0.2
                    ),
                ]
            )
            noise_set = getattr(datasets, self.dataset_name)(
                root=self.data_dir, train=False, transform=noise_transform
            )
            self.noise_loader = dataloader_type(
                dataset=noise_set, batch_size=self.batch_size_test, shuffle=False
            )


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


# @ray.remote
# def run_experiment(name, trainable, exp_config, tune_config):

#     # override when running local for test
#     # if not torch.cuda.is_available():
#     #     exp_config["device"] = "cpu"
#     #     tune_config["resources_per_trial"] = {"cpu": 1}

#     # download dataset
#     download_dataset(exp_config)

#     # run
#     tune_config["name"] = name
#     tune_config["config"] = exp_config
#     tune.run(Trainable, **tune_config)


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
