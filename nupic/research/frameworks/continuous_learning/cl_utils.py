# ----------------------------------------------------------------------
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

import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_multi_model(
    model,
    loader,
    optimizer,
    device,
    freeze_params,
    criterion=F.nll_loss,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    post_batch_callback=None,
    progress_bar=None,

):
    """Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set, or if the number of mini batches
    exceeds the parameter "batches_in_epoch".

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: train dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param batches_in_epoch: Max number of mini batches to train.
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param criterion: loss function to use
    :type criterion: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
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
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(target)
        data = data.to(device, non_blocking=async_gpu)
        target = target.to(device, non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        del data, target, output

        t2 = time.time()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if len(freeze_params) > 0:
            with torch.no_grad():
                for param in freeze_params:
                    param_module = param[0]
                    param_indices = param[1]
                    param_module.grad[param_indices, :] = 0.0

        t3 = time.time()
        optimizer.step()
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3)
            post_batch_callback(model=model, loss=loss.detach(), batch_idx=batch_idx,
                                num_images=num_images, time_string=time_string)
        del loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


def train_model(
    model,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    post_batch_callback=None,
    progress_bar=None,
):
    """Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set, or if the number of mini batches
    exceeds the parameter "batches_in_epoch".

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: train dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param batches_in_epoch: Max number of mini batches to train.
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param criterion: loss function to use
    :type criterion: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
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
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(target)
        data = data.to(device, non_blocking=async_gpu)
        target -= 1
        target = target.to(device, non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        del data, target, output

        t2 = time.time()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        t3 = time.time()
        optimizer.step()
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3)
            post_batch_callback(model=model, loss=loss.detach(), batch_idx=batch_idx,
                                num_images=num_images, time_string=time_string)
        del loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


def unravel_index(index, shape):
    """stole from
    https://github.com/pytorch/pytorch/blob/master/torch/testing/__init__.py
    """
    res = []
    for size in shape[::-1]:
        res.append(int(index % size))
        index = int(index // size)
    return tuple(res[::-1])
