#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
import copy
import os
import socket
from contextlib import closing

import h5py
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DistributedSampler
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop

from nupic.research.frameworks.pytorch.dataset_utils import (
    CachedDatasetFolder,
    HDF5Dataset,
)
from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler

from .auto_augment import ImageNetPolicy

IMAGENET_NUM_CLASSES = {
    10: [
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"
    ],
    100: [
        "n01440764", "n01592084", "n01601694", "n01630670", "n01631663",
        "n01664065", "n01677366", "n01693334", "n01734418", "n01751748",
        "n01755581", "n01855672", "n01877812", "n01978287", "n01981276",
        "n02025239", "n02027492", "n02033041", "n02056570", "n02089867",
        "n02091244", "n02091635", "n02093428", "n02094258", "n02104365",
        "n02105251", "n02106662", "n02107312", "n02108422", "n02112350",
        "n02129165", "n02174001", "n02268443", "n02317335", "n02410509",
        "n02423022", "n02454379", "n02457408", "n02488291", "n02497673",
        "n02536864", "n02640242", "n02655020", "n02727426", "n02783161",
        "n02808304", "n02841315", "n02871525", "n02892201", "n02971356",
        "n02979186", "n02981792", "n03018349", "n03125729", "n03133878",
        "n03207941", "n03250847", "n03272010", "n03372029", "n03400231",
        "n03457902", "n03481172", "n03482405", "n03602883", "n03680355",
        "n03697007", "n03763968", "n03791053", "n03804744", "n03837869",
        "n03854065", "n03891332", "n03954731", "n03956157", "n03970156",
        "n03976657", "n04004767", "n04065272", "n04120489", "n04149813",
        "n04192698", "n04200800", "n04252225", "n04259630", "n04332243",
        "n04335435", "n04346328", "n04350905", "n04404412", "n04461696",
        "n04462240", "n04509417", "n04550184", "n04606251", "n07716358",
        "n07718472", "n07836838", "n09428293", "n13040303", "n15075141"
    ],
}


def create_train_dataloader(
    data_dir, train_dir, batch_size, workers, distributed, num_classes=1000,
    use_auto_augment=False, sample_transform=None, target_transform=None,
    replicas_per_sample=1
):
    """
    Configure Imagenet training dataloader

    Creates :class:`torch.utils.data.DataLoader` using :class:`CachedDatasetFolder`
    or :class:`HDF5Dataset` pre-configured for the training cycle

    :param data_dir: The directory or hdf5 file containing the dataset
    :param train_dir: The directory or hdf5 group containing the training data
    :param batch_size: Images per batch
    :param workers: how many data loading subprocesses to use
    :param distributed: Whether or not to use `DistributedSampler`
    :param num_classes: Limit the dataset size to the given number of classes
    :param sample_transform: List of transforms acting on the samples
                             to be added to the defaults below
    :param target_transform: List of transforms acting on the targets
    :return: torch.utils.data.DataLoader
    """
    if use_auto_augment:
        transform = transforms.Compose(
            transforms=[
                RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    inplace=True
                ),
            ],
        )
    else:
        transform = transforms.Compose(
            transforms=[
                RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    inplace=True
                ),
            ],
        )

    transform = transforms.Compose(
        transforms=[transform] + (sample_transform or []))
    target_transform = target_transform

    if h5py.is_hdf5(data_dir):
        # Use fixed Imagenet classes if mapping is available
        # switch to my temporary version of mixup
        if num_classes in IMAGENET_NUM_CLASSES:
            classes = IMAGENET_NUM_CLASSES[num_classes]
            dataset = HDF5Dataset(hdf5_file=data_dir, root=train_dir,
                                  classes=classes, transform=transform,
                                  target_transform=target_transform,
                                  replicas_per_sample=replicas_per_sample)
        else:
            dataset = HDF5Dataset(hdf5_file=data_dir, root=train_dir,
                                  num_classes=num_classes, transform=transform,
                                  target_transform=target_transform,
                                  replicas_per_sample=replicas_per_sample)
    else:
        dataset = CachedDatasetFolder(root=os.path.join(data_dir, train_dir),
                                      num_classes=num_classes, transform=transform,
                                      target_transform=target_transform)
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


def create_validation_dataloader(data_dir, val_dir, batch_size, workers,
                                 num_classes=1000):
    """
    Configure Imagenet validation dataloader

    Creates :class:`torch.utils.data.DataLoader` using :class:`CachedDatasetFolder`
    or :class:`HDF5Dataset` pre-configured for the validation cycle.

    :param data_dir: The directory or hdf5 file containing the dataset
    :param val_dir: The directory containing or hdf5 group the validation data
    :param batch_size: Images per batch
    :param workers: how many data loading subprocesses to use
    :param num_classes: Limit the dataset size to the given number of classes
    :return: torch.utils.data.DataLoader
    """

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                inplace=True
            ),
        ]
    )
    if h5py.is_hdf5(data_dir):
        if num_classes in IMAGENET_NUM_CLASSES:
            classes = IMAGENET_NUM_CLASSES[num_classes]
            dataset = HDF5Dataset(hdf5_file=data_dir, root=val_dir,
                                  classes=classes, transform=transform)
        else:
            dataset = HDF5Dataset(hdf5_file=data_dir, root=val_dir,
                                  num_classes=num_classes, transform=transform)
    else:
        dataset = CachedDatasetFolder(root=os.path.join(data_dir, val_dir),
                                      num_classes=num_classes, transform=transform)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )


def create_optimizer(model, optimizer_class, optimizer_args,
                     batch_norm_weight_decay, bias_weight_decay):
    """
    Configure the optimizer with the option to ignore `weight_decay` from all
    batch norm module parameters

    :param model:
        The model to get the parameters from

    :param optimizer_class:
        The optimizer class. Must inherit from torch.optim.Optimizer

    :param optimizer_args:
        The optimizer constructor arguments passed in addition to the model parameters

    :param batch_norm_weight_decay:
        Whether or not to apply weight decay to batch norm modules parameters.
        If False, remove 'weight_decay' from batch norm parameters
        See https://arxiv.org/abs/1807.11205

    :param bias_weight_decay:
        Whether or not to apply weight decay to bias modules parameters.
        If False, remove 'weight_decay' from bias parameters

    :return: Configured optimizer
    """

    # get pointers to batch norm layers, don't rely on name
    bn_param_ptrs = set()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            bn_param_ptrs.add(m.weight.data_ptr())
            bn_param_ptrs.add(m.bias.data_ptr())

    group_decay, group_no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.data_ptr() in bn_param_ptrs and not batch_norm_weight_decay:
            group_no_decay.append(param)
        elif ".bias" in name and not bias_weight_decay:
            group_no_decay.append(param)
        else:
            group_decay.append(param)
    model_params = [dict(params=group_decay),
                    dict(params=group_no_decay, weight_decay=.0)]

    return optimizer_class(model_params, **optimizer_args)


def create_lr_scheduler(optimizer, lr_scheduler_class, lr_scheduler_args,
                        steps_per_epoch):
    """
    Configure learning rate scheduler

    :param optimizer:
        Wrapped optimizer
    :param lr_scheduler_class:
        LR scheduler class to use. Must inherit from _LRScheduler
    :param lr_scheduler_args:
        LR scheduler class constructor arguments
    :param steps_per_epoch:
        The total number of batches in the epoch.
        Only used if lr_scheduler_class is :class:`ComposedLRScheduler` or
        :class:`OneCycleLR`
    """
    if issubclass(lr_scheduler_class, OneCycleLR):
        # Update OneCycleLR parameters
        lr_scheduler_args = copy.deepcopy(lr_scheduler_args)
        lr_scheduler_args.update(steps_per_epoch=steps_per_epoch)
    elif issubclass(lr_scheduler_class, ComposedLRScheduler):
        # Update ComposedLRScheduler parameters
        lr_scheduler_args = copy.deepcopy(lr_scheduler_args)
        schedulers = lr_scheduler_args.get("schedulers", None)
        if schedulers is not None:
            # Convert dict from ray/json {str:dict} style to {int:dict}
            schedulers = {int(k): v for k, v in schedulers.items()}

            # Update OneCycleLR "steps_per_epoch" parameter
            for _, item in schedulers.items():
                lr_class = item.get("lr_scheduler_class", None)
                if lr_class is not None and issubclass(lr_class, OneCycleLR):
                    lr_args = item.get("lr_scheduler_args", {})
                    lr_args.update(steps_per_epoch=steps_per_epoch)
            lr_scheduler_args["schedulers"] = schedulers
        lr_scheduler_args["steps_per_epoch"] = steps_per_epoch

    return lr_scheduler_class(optimizer, **lr_scheduler_args)


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # bind on port 0 - kernel will select an unused port
        s.bind(("", 0))
        # removed socket.SO_REUSEADDR arg
        # TCP error due to two process with same rank in same port - maybe a fix
        return s.getsockname()[1]
