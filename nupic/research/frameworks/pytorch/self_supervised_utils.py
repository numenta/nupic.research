# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
import gzip
import io
import pickle
import random
import re
import sys
import time
import warnings
from collections.abc import Collection

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_model_unsupervised(
    model,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    complexity_loss_fn=None,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    post_batch_callback=None,
    pre_model_transform=None,
    transform_to_device_fn=None,
    progress_bar=None,
):
    """Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set, or if the number of mini batches
    exceeds the parameter "batches_in_epoch".

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: unsupervised dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param criterion: loss function to use
    :type criterion: function or nn.Module
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param batches_in_epoch: Max number of mini batches to test on
    :type batches_in_epoch: int
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param pre_model_transform: An optional function which can be used to transform
                                the data before passing into the model. For
                                self-supervised tasks, this would be something like
                                taking an image and turning it into many overlapping
                                patches
    :type pre_model_transform: function
    :param post_model_transform: An optional function which can be used to transform
                                 the output of the model. For self-supervised tasks,
                                 this would be something like taking the final
                                 patch-level representations and aggregating by an
                                 adaptive average pool operation
    :type post_model_transform: function
    :param transform_to_device_fn: Function for sending data and labels to the
                                   device. This provides an extensibility point
                                   for performing any final transformations on
                                   the data or targets, and determining what
                                   actually needs to get sent to the device.
    :type transform_to_device_fn: function
    :param progress_bar: Optional :class:`tqdm` progress bar args.
                         None for no progress bar
    :type progress_bar: dict or None

    :return: mean loss for epoch
    :rtype: float
    """
    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_epoch < len(loader):
            loader.total = batches_in_epoch

    # Check if training with Apex Mixed Precision
    # FIXME: There should be another way to check if 'amp' is enabled
    use_amp = hasattr(optimizer, "_amp_stash")
    try:
        from apex import amp
    except ImportError:
        if use_amp:
            raise ImportError(
                "Mixed precision requires NVIDA APEX."
                "Please install apex from https://www.github.com/nvidia/apex")

    t0 = time.time()
    for batch_idx, (data, _) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(data)
        if transform_to_device_fn is None:
            data = data.to(device, non_blocking=async_gpu)
        else:
            data = transform_to_device_fn(data, device, non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        if pre_model_transform is not None:
            data = pre_model_transform(data)

        optimizer.zero_grad()
        output = model(data)

        error_loss = criterion(output)

        del data, output

        t2 = time.time()
        if use_amp:
            with amp.scale_loss(error_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            error_loss.backward()

        t3 = time.time()

        # Compute and backpropagate the complexity loss. This happens after
        # error loss has backpropagated, freeing its computation graph, so the
        # two loss functions don't compete for memory.
        complexity_loss = (complexity_loss_fn(model)
                           if complexity_loss_fn is not None
                           else None)
        if complexity_loss is not None:
            if use_amp:
                with amp.scale_loss(complexity_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                complexity_loss.backward()

        t4 = time.time()
        optimizer.step()
        t5 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           "complexity loss forward/backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3, t5 - t4)
            post_batch_callback(model=model,
                                error_loss=error_loss.detach(),
                                complexity_loss=(complexity_loss.detach()
                                                 if complexity_loss is not None
                                                 else None),
                                batch_idx=batch_idx,
                                num_images=num_images,
                                time_string=time_string)
        del error_loss, complexity_loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()

def train_model_supervised(
    model,
    classifier,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    batches_in_supervised_training_epoch=sys.maxsize,
    active_classes=None,
    pre_batch_callback=None,
    post_batch_callback=None,
    pre_model_transform=None,
    post_model_transform=None,
    transform_to_device_fn=None,
    progress_bar=None,
):
    """Freezes the weights of the model and trains a classification head on top of
    the frozen model. Trains the given classifier by iterating through mini batches.
    A supervised training epoch ends after one pass through the supervised training
    set, or if the number of mini batches exceeds the parameter
    "batches_in_supervised_training_epoch".

    :param model: a frozen pytorch model
    :type model: torch.nn.Module
    :param classifier: pytorch classifier to be trained
    :param loader: unsupervised dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param criterion: loss function to use
    :type criterion: function or nn.Module
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param batches_in_epoch: Max number of mini batches to test on
    :type batches_in_epoch: int
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param transform_to_device_fn: Function for sending data and labels to the
                                   device. This provides an extensibility point
                                   for performing any final transformations on
                                   the data or targets, and determining what
                                   actually needs to get sent to the device.
    :type transform_to_device_fn: function
    :param progress_bar: Optional :class:`tqdm` progress bar args.
                         None for no progress bar
    :type progress_bar: dict or None

    :return: mean loss for epoch
    :rtype: float
    """


    # Freeze the original model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_supervised_training_epoch < len(loader):
            loader.total = batches_in_supervised_training_epoch

    # Check if training with Apex Mixed Precision
    # FIXME: There should be another way to check if 'amp' is enabled
    use_amp = hasattr(optimizer, "_amp_stash")
    try:
        from apex import amp
    except ImportError:
        if use_amp:
            raise ImportError(
                "Mixed precision requires NVIDA APEX."
                "Please install apex from https://www.github.com/nvidia/apex")

    t0 = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_supervised_training_epoch:
            break

        num_images = len(target)
        if transform_to_device_fn is None:
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)
        else:
            data, target = transform_to_device_fn(data, target, device,
                                                  non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        if pre_model_transform is not None:
            data = pre_model_transform(data)
        #we don't want to calculate gradients during supervised training step
        with torch.no_grad():
            output = model(data)
            if post_model_transform is not None:
                output = post_model_transform(output)


        optimizer.zero_grad()

        output = classifier(output)
        if active_classes is not None:
            output = output[:, active_classes]
        error_loss = criterion(output)

        del data, target, output

        t2 = time.time()
        if use_amp:
            with amp.scale_loss(error_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            error_loss.backward()

        t3 = time.time()
        optimizer.step()
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3)
            post_batch_callback(model=model,
                                error_loss=error_loss.detach(),
                                batch_idx=batch_idx,
                                num_images=num_images,
                                time_string=time_string)
        del error_loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


def evaluate_model(
    model,
    classifier,
    loader,
    device,
    batches_in_epoch=sys.maxsize,
    criterion=F.nll_loss,
    active_classes=None,
    progress=None,
    pre_batch_callback=None,
    post_batch_callback=None,
    pre_model_transform=None,
    post_model_transform=None,
    transform_to_device_fn=None,
):
    """Evaluate pre-trained model using given test dataset loader.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param classifier: Pretrained pytorch classifier model
    :type classifier: torch.nn.Module
    :param loader: test dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device`
    :param batches_in_epoch: Max number of mini batches to test on
    :type batches_in_epoch: int
    :param criterion: loss function to use
    :type criterion: function
    :param active_classes: a list of indices of the heads that are active for a given
                           task; only relevant if this function is being used in a
                           continual learning scenario
    :type active_classes: list of int or None
    :param progress: Optional :class:`tqdm` progress bar args. None for no progress bar
    :type progress: dict or None
    :param pre_batch_callback:
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters:
                                batch_idx, target, output, pred
    :type post_batch_callback: function
    :param transform_to_device_fn: Function for sending data and labels to the
                                   device. This provides an extensibility point
                                   for performing any final transformations on
                                   the data or targets, and determining what
                                   actually needs to get sent to the device.
    :type transform_to_device_fn: function

    :return: dictionary with computed "mean_accuracy", "mean_loss", "total_correct".
    :rtype: dict
    """

    model.eval()
    classifier.eval()
    total = 0

    # Perform accumulation on device, avoid paying performance cost of .item()
    loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)

    async_gpu = loader.pin_memory

    if progress is not None:
        loader = tqdm(loader, total=min(len(loader), batches_in_epoch),
                      **progress)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= batches_in_epoch:
                break

            if transform_to_device_fn is None:
                data = data.to(device, non_blocking=async_gpu)
                target = target.to(device, non_blocking=async_gpu)
            else:
                data, target = transform_to_device_fn(data, target, device,
                                                      non_blocking=async_gpu)

            if pre_batch_callback is not None:
                pre_batch_callback(model=model, batch_idx=batch_idx)

            if pre_model_transform is not None:
                data = pre_model_transform(data)

            output = model(data)

            if post_model_transform is not None:
                output = post_model_transform(output)

            output = classifier(output)
            if active_classes is not None:
                output = output[:, active_classes]
            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

            if post_batch_callback is not None:
                post_batch_callback(batch_idx=batch_idx, target=target, output=output,
                                    pred=pred)

    if progress is not None:
        loader.close()

    correct = correct.item()
    loss = loss.item()

    result = {
        "total_correct": correct,
        "total_tested": total,
        "mean_loss": loss / total if total > 0 else 0,
        "mean_accuracy": correct / total if total > 0 else 0,
    }

    return result

