# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

"""
Functions for training and testing a network with single/multiple heads in a continuous
learning framework.

This implementation is based on the original continual learning benchmarks repository:
https://github.com/GMvandeVen/continual-learning
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from nupic.research.frameworks.pytorch.dataset_utils.dataset_utils import split_dataset


def train_head(model, loader, optimizer, criterion, device, active_classes=None,
               post_batch_callback=None):
    """
    Train the model on a single task using given dataset loader and set of active
    classes.
    Called on every epoch.
    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
    :type optimizer: :class:`torch.optim.Optimizer`
    :param criterion: loss function to use
    :type criterion: function
    :param active_classes: list of int specifying the "active" classes
    :type active_classes: list or None
    :param device:
    :type device: :class:`torch.device`
    :param post_batch_callback: function(model) to call after every batch
    :type post_batch_callback: function
    """
    model.train()
    for data, target in tqdm(loader, desc="Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # compute loss only for 'active' output units, and only backpropogate
        # errors for these units when calling loss.backward()
        output = active_class_outputs(model, data, active_classes)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if post_batch_callback is not None:
            post_batch_callback(model)


def test(model, loader, criterion, device, allowed_classes=None):
    """
    Evaluate trained model on a single task using given dataset loader and set of
    "allowed" classes.
    Called on every epoch.
    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: DataLoader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param criterion: loss function to use
    :type criterion: function
    :param device:
    :type device: :class:`torch.device`
    :param allowed_classes: list of int specifying the "active" classes
    :type allowed_classes: list or None
    :param desc: Description for progress bar
    :type desc: str
    :return: Dict with "accuracy", "loss" and "total_correct"
    """
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Test", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)

            if allowed_classes is not None:
                output = output[:, allowed_classes]

            # sum up batch loss
            loss += criterion(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}


def do_training(model, scenario, device, lr=0.001, epochs=30,
                train_batch_size=64, test_batch_size=1000, post_batch_callback=None):
    """
    Train the model.
    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param scenario: continuous learning setup, one of {'task', 'domain', 'class'}
    :type scenario: str
    :param device:
    :type device: torch.device
    :param lr: learning rate for the optimizer
    :type lr: float
    :param epochs: number of epochs for which to train the classifier
    :type epochs: int
    :param train_batch_size: batch size during training
    :type train_batch_size: int
    :param test_batch_size: batch size during training
    :type test_batch_size: int
    :param post_batch_callback: function(model) to call after every batch
    :type post_batch_callback: function
    """
    train_mnist = MNIST(root=".", train=True, transform=transforms.ToTensor())
    test_mnist = MNIST(root=".", train=True, transform=transforms.ToTensor())

    train_datasets = split_dataset(train_mnist,
                                   groupby=lambda x: x[1] // 2)
    test_datasets = split_dataset(test_mnist,
                                  groupby=lambda x: x[1] // 2)

    # apply transformation to target variables, depending on the scenario
    target_transform = get_target_transform(scenario)

    if target_transform is not None:
        for train_set in train_datasets:
            train_set.dataset.targets = target_transform(train_set.dataset.targets)

        for test_set in test_datasets:
            test_set.dataset.targets = target_transform(test_set.dataset.targets)

    # data loaders
    train_loaders = [
        torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle=True)
        for train_dataset in train_datasets]
    test_loaders = [
        torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                    shuffle=True)
        for test_dataset in test_datasets]

    n_tasks = len(train_datasets)

    # optimizer for training model
    adam = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epochs):

        # Loop over all tasks
        for task_num, train_loader in enumerate(train_loaders, 1):

            print("Epoch {}".format(epoch))
            train_head(model=model, loader=train_loader, optimizer=adam,
                       criterion=F.nll_loss, device=device,
                       active_classes=get_active_classes(task_num, scenario),
                       post_batch_callback=post_batch_callback)

            # test model
            results = []
            for i, _allowed_classes in enumerate([[0, 1], [2, 3], [4, 5], [6, 7],
                                                 [8, 9]]):

                allowed_classes = _allowed_classes if scenario == "task" else None
                results = test(model=model, loader=test_loaders[i],
                               criterion=F.nll_loss, device=device,
                               allowed_classes=allowed_classes)
                print("\ttask {}/{}: {}".format(i + 1, n_tasks, results))

            print("Epoch {}: {}".format(epoch, results))


def get_target_transform(scenario):
    """
    Returns the appropriate pytorch transform to apply to dataset target variables
    based on the continual learning scenario.
    :param scenario: continuous learning setup, one of {'task', 'domain', 'class'}
    :type scenario: str
    """
    if scenario in ("task", "domain"):
        target_transform = transforms.Lambda(lambda y: y % 2)
    else:
        target_transform = None
    return target_transform


def active_class_outputs(model, inputs, active_classes):
    """
    Computes the output using the given model and input data, and and returns the
    output for "active" classes only.
    :param model: pytorch model to be used to compute outputs
    :type model: torch.nn.Module
    :param inputs: input batch to model
    :type inputs: pytorch tensor
    :param active_classes: list of int specifying the "active" classes
    :type active_classes: list of int
    """
    output = model(inputs)
    if active_classes is not None:
        return output[:, active_classes]
    else:
        return output


def get_active_classes(task_num, scenario):
    """
    Return a list of label indices that are "active" during training under the given
    scenario. In the <task> scenario, only the classes that are being trained are
    active. In the <domain> scenario, there is only one output head, so the concept of
    "active" classes is not relevant. In the <class> scenario, all tasks that the model
    has previously observed are active. More information about active classes can be
    found here: https://arxiv.org/abs/1904.07734

    :param task_num: the index of the task
    :type task_num: int
    :param scenario: continuous learning setup, one of {'task', 'domain', 'class'}
    :type scenario: str
    :param device:
    :type device: torch.device
    :param post_batch_callback: function(model) to call after every batch
    :type post_batch_callback: function
    :return: list of int
    """
    if scenario == "task":
        return [2 * task_num - 2, 2 * task_num - 1]
    elif scenario == "class":
        return [label for label in range(2 * task_num)]
    elif scenario == "domain":
        return None
    raise Exception("Argument `scenario` must be 'task', 'domain', or 'class'")
