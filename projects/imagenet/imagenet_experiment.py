#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import bisect
import copy
import functools
import io
import itertools
import logging
import os
import pickle
import sys

import ray.services
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DistributedSampler
from torchvision import transforms
from torchvision.models.resnet import BasicBlock, Bottleneck

from nupic.research.frameworks.pytorch.dataset_utils import CachedDatasetFolder
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.torch.modules import rezero_weights, update_boost_strength

__all__ = ["ImagenetExperiment"]

# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class ProgressiveRandomResizedCrop(transforms.Compose):
    """
    Progressive resize and crop image transform. This transform will apply a
    different size to `torchvision.transforms.RandomResizedCrop` based on the
    epoch. The method `set_epoch` must be called on each epoch before using
    this transform

    :param epoch_resize: A dictionary mapping epoch to image size. Each key
                         correspond to the start epoch and the value correspond
                         to the image size. For example:
                         epoch_resize={
                             0: 128, # epoch  0-13,  resize to 128
                            14: 224, # epoch 14-31,  resize to 224
                            32: 288, # epoch 32-end, resize to 288
                        },

    :param transforms: List of transforms to apply after the image is resized
    """

    def __init__(self, epoch_resize, transforms):
        super().__init__(transforms)
        self.epoch_resize = epoch_resize
        self.resize = None

    def __call__(self, img):
        img = self.resize(img)
        return super().__call__(img)

    def set_epoch(self, epoch):
        key = bisect.bisect(sorted(self.epoch_resize.keys()), epoch) - 1
        self.resize = transforms.RandomResizedCrop(self.epoch_resize[key])


def _create_train_dataloader(
    data_dir, batch_size, workers, distributed, epoch_resize, num_classes=1000
):
    """
    Configure Imagenet training dataloader

    :param data_dir: The directory containing the training data
    :param batch_size: Images per batch
    :param workers: how many data loading subprocesses to use
    :param distributed: Whether or not to use `DistributedSampler`
    :param epoch_resize: Dictionary containing the progressive resize schedule
    :param num_classes: Limit the dataset size to the given number of classes
    :return: torch.utils.data.DataLoader
    """
    dataset = CachedDatasetFolder(
        root=data_dir,
        num_classes=num_classes,
        transform=ProgressiveRandomResizedCrop(
            epoch_resize=epoch_resize,
            transforms=[
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ],
        ),
    )
    if distributed:
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        num_workers=workers,
        sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
    )


def _create_validation_dataloader(data_dir, batch_size, workers, num_classes=1000):
    """
    Configure Imagenet validation dataloader

    :param data_dir: The directory containing the validation data
    :param batch_size: Images per batch
    :param workers: how many data loading subprocesses to use
    :param num_classes: Limit the dataset size to the given number of classes
    :return: torch.utils.data.DataLoader
    """
    dataset = CachedDatasetFolder(
        root=data_dir,
        num_classes=num_classes,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )


def _create_optimizer(model, optimizer_class, optimizer_args, optimizer_groups=None):
    """
    Configure the optimizer with or without parameter groups

    :param model:
        The model to get the parameters from

    :param optimizer_class:
        The optimizer class. Must inherit from torch.optim.Optimizer

    :param optimizer_args:
        The optimizer constructor arguments passed in addition to the model parameters

    :param optimizer_groups: Optional dictionary used to customize the optimizer
        parameters (lr, wd, ...) for each model parameter group. This dictionary
        contains the following fields:
        - "group_by": key/func used to group the model submodules via itertools.groupby
        - "parameters": dict of optimizer parameters to override for each group.

      For example, use the following values to remove "weight_decay" from
      "BatchNorm" module::

        optimizer_groups = {
            "group_by": lambda module: isinstance(module, BatchNorm),
            "parameters": {
                "True": {"weight_decay": 0.},  # BatchNorm modules
                "False": {},                   # All other modules
            }
        }

    :return: Configured optimizer
    """
    # Create custom parameter groups
    if optimizer_groups is not None:
        # Sort modules using "group_by" key
        group_by = optimizer_groups["group_by"]
        sorted_modules = sorted(model.modules(), key=group_by)

        # Group module parameters using "group_by" key
        grouped_parameters = {
            str(k): list(itertools.chain.from_iterable(m.parameters(False) for m in g))
            for k, g in itertools.groupby(sorted_modules, key=group_by)
        }

        # Add custom optimizer parameters for each group
        model_params = []
        parameters = optimizer_groups["parameters"]
        for k, params in grouped_parameters.items():
            group = copy.deepcopy(optimizer_args)
            group.update(params=params)
            group.update(**parameters[k])
            model_params.append(group)
    else:
        model_params = model.parameters()

    return optimizer_class(model_params, **optimizer_args)


def _create_model(model_class, model_args, init_batch_norm, distributed, device):
    """
    Configure network model

    :param model_class:
            The model class. Must inherit from torch.nn.Module
    :param model_args:
        The model constructor arguments
    :param init_batch_norm:
        Whether or not to initialize running batch norm mean to 0
        See https://arxiv.org/pdf/1706.02677.pdf
    :param distributed:
        Whether or not to use `DistributedDataParallel`
    :param device:
        Model device

    :return: Configured model
    """
    model = model_class(**model_args)
    model.to(device)
    if distributed:
        model = DistributedDataParallel(model)
    else:
        model = DataParallel(model)

    # See https://arxiv.org/pdf/1706.02677.pdf
    if init_batch_norm:
        for m in model.modules():
            if isinstance(m, BasicBlock):
                # initialized the last BatchNorm in each BasicBlock to 0
                m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
            elif isinstance(m, Bottleneck):
                # initialized the last BatchNorm in each Bottleneck to 0
                m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
            elif isinstance(m, nn.Linear):
                # initialized linear layers weights from a gaussian distribution
                m.weight.data.normal_(0, 0.01)

    return model


class ImagenetExperiment:
    """
    Experiment class used to train different models on Imagenet
    """

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batches_in_epoch = sys.maxsize
        self.steps_per_epoch = 0
        self.epochs = 1
        self.distributed = False
        self.rank = 0
        self.total_batches = 0
        self.progress = False
        self.logger = None

    def setup_experiment(self, config):
        """
        Configure the experiment for training
        :param config: Dictionary containing the configuration parameters
            - distributed: Whether or not to use  Pytorch Distributed training
            - backend: Pytorch Distributed backend ("nccl", "gloo")
            - world_size: Total number of processes participating
            - rank: Rank of the current process

            - data: Dataset path
            - train_dir: Dataset training data relative path
            - batch_size: Training batch size
            - val_dir: Dataset validation data relative path
            - val_batch_size: Validation batch size
            - workers: how many data loading processes to use
            - num_classes: Limit the dataset size to the given number of classes

            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model model class arguments passed to the constructor
            - init_batch_norm: Whether or not to Initialize running batch norm
                               mean to 0.

            - optimizer_class: Optimizer class.
                               Must inherit from "torch.optim.Optimizer"
            - optimizer_args: Optimizer class class arguments passed to the
                              constructor
            - weight_decay_batch_norm: Whether or not to apply weight decay to
                                       batch norm modules parameters

            - lr_scheduler_class: Learning rate scheduler class.
                                 Must inherit from "_LRScheduler"
            - lr_scheduler_args: Learning rate scheduler class class arguments
                                 passed to the constructor

            - loss_function: Loss function. See "torch.nn.functional"
            - local_dir: Results path
            - epochs: Number of epochs to train
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - progress: Show progress during training
            - name: Experiment name. Used as logger name
            - log_level: Python Logging level
            - log_format: Python Logging format
        """
        # Configure logger
        log_format = config.get("log_format", logging.BASIC_FORMAT)
        log_level = getattr(logging, config.get("log_level", "INFO").upper())
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger(config.get("name", type(self).__name__))
        self.logger.setLevel(log_level)
        self.logger.addHandler(console)
        self.progress = config.get("progress", False)

        # Configure distribute pytorch
        self.distributed = config.get("distributed", False)
        self.rank = config.get("rank", 0)
        if self.distributed:
            dist_url = config.get("dist_url", "tcp://127.0.0.1:54321")
            backend = config.get("backend", "nccl")
            world_size = config.get("world_size", 1)
            dist.init_process_group(
                backend=backend,
                init_method=dist_url,
                rank=self.rank,
                world_size=world_size,
            )

        # Configure data loaders
        self.epochs = config.get("epochs", 1)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        workers = config.get("workers", 0)
        train_dir = os.path.join(config["data"], config.get("train_dir", "train"))
        batch_size = config.get("batch_size", 1)
        epoch_resize = config.get("epoch_resize", {0: 224})
        num_classes = config.get("num_classes", 1000)
        self.train_loader = _create_train_dataloader(
            data_dir=train_dir,
            batch_size=batch_size,
            workers=workers,
            distributed=self.distributed,
            epoch_resize=epoch_resize,
            num_classes=num_classes,
        )
        self.steps_per_epoch = len(self.train_loader)

        val_dir = os.path.join(config["data"], config.get("val_dir", "val"))
        val_batch_size = config.get("val_batch_size", batch_size)
        self.val_loader = _create_validation_dataloader(
            data_dir=val_dir,
            batch_size=val_batch_size,
            workers=workers,
            num_classes=num_classes,
        )

        # Configure model
        model_class = config["model_class"]
        model_args = config.get("model_args", {})
        init_batch_norm = config.get("init_batch_norm", False)
        self.model = _create_model(
            model_class=model_class,
            model_args=model_args,
            init_batch_norm=init_batch_norm,
            distributed=self.distributed,
            device=self.device,
        )

        # Configure optimizer
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        optimizer_groups = None

        # Remove weight decay from batch norm modules
        if not config.get("weight_decay_batch_norm", True):
            # Group parameters by "BatchNorm" and not "BatchNorm"
            optimizer_groups = dict(
                group_by=lambda module: isinstance(module, _BatchNorm),
                parameters={
                    # Remove 'weight_decay' from _BatchNorm parameters
                    "True": dict(weight_decay=0.0),
                    # Leave all other parameters alone
                    "False": {},
                },
            )

        self.optimizer = _create_optimizer(
            model=self.model,
            optimizer_class=optimizer_class,
            optimizer_args=optimizer_args,
            optimizer_groups=optimizer_groups,
        )

        self.loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )

        self.total_batches = len(self.train_loader)
        # Configure leaning rate scheduler
        lr_scheduler_class = config.get("lr_scheduler_class", None)
        if lr_scheduler_class is not None:
            lr_scheduler_args = config.get("lr_scheduler_args", {})
            if lr_scheduler_class == OneCycleLR:
                lr_scheduler_args = copy.deepcopy(lr_scheduler_args)
                lr_scheduler_args["epochs"] = self.epochs
                lr_scheduler_args["steps_per_epoch"] = self.steps_per_epoch

            self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_args)

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        results = evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.loss_function,
            batches_in_epoch=self.batches_in_epoch,
        )
        results.update(
            learning_rate=self.get_lr()[0],
        )
        self.logger.info(results)

        return results

    def train_epoch(self, epoch):

        train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.loss_function,
            batches_in_epoch=self.batches_in_epoch,
            pre_batch_callback=functools.partial(self.pre_batch, epoch=epoch),
            post_batch_callback=functools.partial(self.post_batch, epoch=epoch),
        )

    def run_epoch(self, epoch):
        self.pre_epoch(epoch)
        self.train_epoch(epoch)
        self.post_epoch(epoch)

        return self.validate()

    def pre_epoch(self, epoch):
        self.model.apply(update_boost_strength)
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

        # Update transform
        transform = self.train_loader.dataset.transform
        if isinstance(transform, ProgressiveRandomResizedCrop):
            transform.set_epoch(epoch)

    def pre_batch(self, model, batch_idx, epoch):
        pass

    def post_batch(self, model, loss, batch_idx, epoch):
        if self.progress and self.rank == 0 and (batch_idx % 10) == 0:
            total_batches = self.total_batches
            current_batch = batch_idx
            if self.distributed:
                # Compute actual batch size from distributed sampler
                total_batches *= self.train_loader.sampler.num_replicas
                current_batch *= self.train_loader.sampler.num_replicas
            self.logger.info("Epoch: %s, Batch: %s/%s, loss: %s",
                             epoch, current_batch, total_batches, loss)

        if isinstance(self.lr_scheduler, OneCycleLR):
            step = epoch * self.steps_per_epoch + batch_idx + 1
            self.lr_scheduler.step(step)

    def post_epoch(self, epoch):
        self.model.apply(rezero_weights)
        if not isinstance(self.lr_scheduler, OneCycleLR):
            self.lr_scheduler.step(epoch)

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        # Save state into a byte array to avoid ray's GPU serialization issues
        # See https://github.com/ray-project/ray/issues/5519
        state = {}
        with io.BytesIO() as buffer:
            torch.save(
                self.model.module.state_dict(),
                buffer,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
            state["model"] = buffer.getvalue()

        with io.BytesIO() as buffer:
            torch.save(
                self.optimizer.state_dict(),
                buffer,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
            state["optimizer"] = buffer.getvalue()

        with io.BytesIO() as buffer:
            torch.save(
                self.lr_scheduler.state_dict(),
                buffer,
                pickle_protocol=pickle.HIGHEST_PROTOCOL,
            )
            state["lr_scheduler"] = buffer.getvalue()

        return state

    def set_state(self, state):
        """
        Restore the experiment from the state returned by `get_state`
        :param state: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        if "model" in state:
            with io.BytesIO(state["model"]) as buffer:
                state_dict = torch.load(buffer, map_location=self.device)
                self.model.module.load_state_dict(state_dict)

        if "optimizer" in state:
            with io.BytesIO(state["optimizer"]) as buffer:
                state_dict = torch.load(buffer, map_location=self.device)
                self.optimizer.load_state_dict(state_dict)

        if "lr_scheduler" in state:
            with io.BytesIO(state["lr_scheduler"]) as buffer:
                state_dict = torch.load(buffer, map_location=self.device)
                self.lr_scheduler.load_state_dict(state_dict)

    def stop_experiment(self):
        if self.distributed:
            dist.destroy_process_group()

    def get_lr(self):
        """
        Returns the current learning rate
        :return: list of learning rates used by the optimizer
        """
        return [p["lr"] for p in self.optimizer.param_groups]

    def get_node_ip(self):
        """Returns the IP address of the current ray node."""
        return ray.services.get_node_ip_address()
