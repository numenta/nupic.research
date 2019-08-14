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
import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nupic.torch.modules import KWinners, KWinners2d, SparseWeights, SparseWeights2d

logging.basicConfig(level=logging.ERROR)


def train_model(
    model,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    batches_in_epoch=sys.maxsize,
    batch_callback=None,
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
    :param batch_callback: Callback function to be called on every batch with the
                           following parameters: model, batch_idx
    :type batch_callback: function
    :param progress_bar: Optional :class:`tqdm` progress bar args.
                         None for no progress bar
    :type progress_bar: dict or None
    """
    model.train()
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_epoch < len(loader):
            loader.total = batches_in_epoch

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_callback is not None:
            batch_callback(model=model, batch_idx=batch_idx)

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


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
    dataset_len = len(loader.sampler)

    if progress is not None:
        loader = tqdm(loader, **progress)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= batches_in_epoch:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    if progress is not None:
        loader.close()

    return {
        "total_correct": correct,
        "mean_loss": loss / dataset_len,
        "mean_accuracy": correct / dataset_len,
    }


def set_random_seed(seed):
    """Set pytorch random seed.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def add_sparse_cnn_layer(
    network,
    suffix,
    in_channels,
    out_channels,
    use_batch_norm,
    weight_sparsity,
    percent_on,
    k_inference_factor,
    boost_strength,
    boost_strength_factor,
):
    """Add sparse cnn layer to network.

    :param network: The network to add the sparse layer to
    :param suffix: Layer suffix. Used to name its components
    :param in_channels: input channels
    :param out_channels: output channels
    :param use_batch_norm: whether or not to use batch norm
    :param weight_sparsity: Pct of weights that are allowed to be non-zero
    :param percent_on: Pct of ON (non-zero) units
    :param k_inference_factor: During inference we increase percent_on by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor:
        boost strength is multiplied by this factor after each epoch
    """
    cnn = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=5,
        padding=0,
        stride=1,
    )
    if 0 < weight_sparsity < 1.0:
        sparse_cnn = SparseWeights2d(cnn, weight_sparsity)
        network.add_module("cnnSdr{}_cnn".format(suffix), sparse_cnn)
    else:
        network.add_module("cnnSdr{}_cnn".format(suffix), cnn)

    if use_batch_norm:
        bn = nn.BatchNorm2d(out_channels, affine=False)
        network.add_module("cnnSdr{}_bn".format(suffix), bn)

    # Max pool
    maxpool = nn.MaxPool2d(kernel_size=2)
    network.add_module("cnnSdr{}_maxpool".format(suffix), maxpool)

    if 0 < percent_on < 1.0:
        kwinner = KWinners2d(
            channels=out_channels,
            percent_on=percent_on,
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
        )
        network.add_module("cnnSdr{}_kwinner".format(suffix), kwinner)
    else:
        network.add_module("cnnSdr{}_relu".format(suffix), nn.ReLU())


def add_sparse_linear_layer(
    network,
    suffix,
    input_size,
    linear_n,
    dropout,
    use_batch_norm,
    weight_sparsity,
    percent_on,
    k_inference_factor,
    boost_strength,
    boost_strength_factor,
):
    """Add sparse linear layer to network.

    :param network: The network to add the sparse layer to
    :param suffix: Layer suffix. Used to name its components
    :param input_size: Input size
    :param linear_n: Number of units
    :param dropout: dropout value
    :param use_batch_norm: whether or not to use batch norm
    :param weight_sparsity: Pct of weights that are allowed to be non-zero
    :param percent_on: Pct of ON (non-zero) units
    :param k_inference_factor: During inference we increase percent_on by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor:
        boost strength is multiplied by this factor after each epoch
    """
    linear = nn.Linear(input_size, linear_n)
    if 0 < weight_sparsity < 1.0:
        network.add_module(
            "linear{}".format(suffix), SparseWeights(linear, weight_sparsity)
        )
    else:
        network.add_module("linear{}".format(suffix), linear)

    if use_batch_norm:
        network.add_module("linear_bn", nn.BatchNorm1d(linear_n, affine=False))

    if dropout > 0.0:
        network.add_module("linear{}_dropout".format(suffix), nn.Dropout(dropout))

    if 0 < percent_on < 1.0:
        network.add_module(
            "linear{}_kwinners".format(suffix),
            KWinners(
                n=linear_n,
                percent_on=percent_on,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
            ),
        )

    else:
        network.add_module("linear{}_relu".format(suffix), nn.ReLU())
