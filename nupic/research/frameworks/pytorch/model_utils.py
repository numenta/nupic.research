# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break
        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        data = data.to(device, non_blocking=async_gpu)
        target = target.to(device, non_blocking=async_gpu)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if progress_bar:
            loader.set_postfix(dict(loss=loss.item()))

        if post_batch_callback is not None:
            post_batch_callback(model=model, loss=loss.item(), batch_idx=batch_idx)

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()

    return total_loss / len(loader)


def evaluate_model(
    model,
    loader,
    device,
    batches_in_epoch=sys.maxsize,
    criterion=F.nll_loss,
    progress=None,
):
    """Evaluate pre-trained model using given test dataset loader.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: test dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device`
    :param batches_in_epoch: Max number of mini batches to test on.
    :type batches_in_epoch: int
    :param criterion: loss function to use
    :type criterion: function
    :param progress: Optional :class:`tqdm` progress bar args. None for no progress bar
    :type progress: dict or None

    :return: dictionary with computed "mean_accuracy", "mean_loss", "total_correct".
    :rtype: dict
    """
    model.eval()
    loss = 0
    correct = 0
    total = 0
    async_gpu = loader.pin_memory

    if progress is not None:
        loader = tqdm(loader, **progress)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= batches_in_epoch:
                break
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)

            output = model(data)
            loss += criterion(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            if progress:
                loader.set_postfix(dict(
                    acc=correct / total,
                    loss=loss / total))

    if progress is not None:
        loader.close()

    return {
        "total_correct": correct,
        "mean_loss": loss / total if total > 0 else 0,
        "mean_accuracy": correct / total if total > 0 else 0,
    }


def set_random_seed(seed, deterministic_mode=True):
    """
    Set pytorch, python random, and numpy random seeds (these are all the seeds we
    normally use).

    :param seed:  (int) seed value
    :param deterministic_mode: (bool) If True, then even on a GPU we'll get more
           deterministic results, though performance may be slower. See:
           https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available() and deterministic_mode:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_nonzero_params(model):
    """
    Count the total number of non-zero weights in the model, including bias weights.
    """
    total_nonzero_params = 0
    total_params = 0
    for param in model.parameters():
        total_nonzero_params += param.data.nonzero().size(0)
        total_params += param.data.numel()

    return total_params, total_nonzero_params
