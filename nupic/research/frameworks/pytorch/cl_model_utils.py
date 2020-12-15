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
This module is similar to model_utils.py, except specific for training and evaluating
in continual learning scenarios.
"""

import time

import torch
import torch.nn.functional as F


def train_cl_model(
    model,
    scenario,
    num_classes_per_task,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    pre_batch_callback=None,
    post_batch_callback=None
):
    """
    Train the given model in a specified continual learning scenario by iterating
    through mini batches. This function specifically designed to deal with multiple
    output heads.

    :param model: pytorch model to be trained
    :param scenario: continual learning scenario, one of "task", "domain", or "class"
    :param num_classes_per_task: number of classes per task
    :param loader: train dataset loader
    :param optimizer: Optimizer object used to train the model. This function will
                      train the model on every batch using this optimizer and the
                      :func:`torch.nn.functional.nll_loss` function
    :param device: device to use ('cpu' or 'cuda')
    :param criterion: loss function to use
    :param pre_batch_callback: Callback function to be called before every batch with
                               the following parameters: model, batch_idx
    :param post_batch_callback: Callback function to be called after every batch with
                                the following parameters: model, batch_idx
    """
    model.train()
    async_gpu = loader.pin_memory
    t0 = time.time()

    for batch_idx, (data, target) in enumerate(loader):

        num_images = len(target)
        data = data.to(device, non_blocking=async_gpu)
        target = target.to(device, non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()

        # Compute loss only for "active" output units, and only backpropogate errors
        # for these units when calling loss.backward()
        active_classes = get_active_classes(scenario,
                                            loader.sampler.active_tasks[-1],
                                            num_classes_per_task)
        output = model(data)
        output = output[:, active_classes]
        error_loss = criterion(output, target)
        t2 = time.time()

        error_loss.backward()
        t3 = time.time()

        optimizer.step()
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1,
                                                              t3 - t2, t4 - t3)
            post_batch_callback(model=model,
                                error_loss=error_loss.detach(),
                                complexity_loss=None,
                                batch_idx=batch_idx,
                                num_images=num_images,
                                time_string=time_string)
        del error_loss
        t0 = time.time()


def evaluate_cl_model(
    model,
    scenario,
    num_classes_per_task,
    loader,
    device,
    criterion=F.nll_loss,
    post_batch_callback=None,
):
    """
    Evaluates a given model similar to `evaluate_model` used by
    `SupervisedExperiment`, but this function is compatible with the "task" and
    "class" continual learning scenarios.

    :param model: pytorch model to be trained
    :param scenario: continual learning scenario, one of "task", "domain", or "class"
    :param num_classes_per_task: number of classes per task
    :param loader: train dataset loader
    :param device: device to use ('cpu' or 'cuda')
    :param criterion: loss function to use
    :param post_batch_callback: Callback function to be called after every batch with
                                the following parameters: model, batch_idx
    """
    model.eval()
    total = 0

    # Perform accumulation on device, avoid paying performance cost of .item()
    loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)

    async_gpu = loader.pin_memory

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)

            # Compute loss only for "active" output units, and only backpropogate
            # errors for these units when calling loss.backward()
            active_classes = get_active_classes(scenario,
                                                loader.sampler.active_tasks[-1],
                                                num_classes_per_task)
            output = model(data)
            output = output[:, active_classes]

            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

            if post_batch_callback is not None:
                post_batch_callback(batch_idx=batch_idx, target=target,
                                    output=output, pred=pred)

    correct = correct.item()
    loss = loss.item()

    result = {
        "total_correct": correct,
        "total_tested": total,
        "mean_loss": loss / total if total > 0 else 0,
        "mean_accuracy": correct / total if total > 0 else 0,
    }

    return result


def get_active_classes(scenario, task_num, num_classes_per_task):
    """
    Returns a list of label indices that are "active" during training under the
    continual learning scenario specified by the config. In the "task" scenario, only
    the classes that are being trained are active. In the "class" scenario, all tasks
    that the model has previously observed are active. More information about active
    classes can be found here: https://arxiv.org/abs/1904.07734

    :param task_num: zero-based index of the task
    :param scenario: continual learning scenario, one of "task", "domain", or "class"
    :param num_classes_per_task: the number of classes per task
    """
    if scenario == "task":
        return [label for label in range(
            num_classes_per_task * task_num,
            num_classes_per_task * (task_num + 1)
        )]

    elif scenario == "domain":
        # TODO add domain scenario
        raise NotImplementedError

    elif scenario == "class":
        return [label for label in range(
            num_classes_per_task * (task_num + 1)
        )]

    else:
        raise Exception("`scenario` must be either 'task', 'domain', or 'class'")
