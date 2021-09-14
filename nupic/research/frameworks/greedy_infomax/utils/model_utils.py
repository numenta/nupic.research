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
#
# This work was based on the original Greedy InfoMax codebase from Sindy Lowe:
# https://github.com/loeweX/Greedy_InfoMax
# The Greedy InfoMax paper can be found here:
# https://arxiv.org/abs/1905.11786
# ----------------------------------------------------------------------

import sys
import time

import torch
import torch.nn.functional as F


def gen_orthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def make_delta_orthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = gen_orthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)


def patchify_inputs(x, patch_size, overlap):
    x = (
        x.unfold(2, patch_size, patch_size // overlap)
        .unfold(3, patch_size, patch_size // overlap)
        .permute(0, 2, 3, 1, 4, 5)  # b, p_x, p_y, c, x, y
    )
    n_patches_x = x.shape[1]
    n_patches_y = x.shape[2]
    x = x.reshape(
        x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
    )
    return x, n_patches_x, n_patches_y


def train_gim_model(
    model,
    loader,
    optimizer,
    device,
    criterion=F.nll_loss,
    complexity_loss_fn=None,
    batches_in_epoch=sys.maxsize,
    active_classes=None,
    pre_batch_callback=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
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
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param criterion: loss function to use
    :type criterion: function
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param batches_in_epoch: Max number of mini batches to test on
    :type batches_in_epoch: int
    :param active_classes: a list of indices of the heads that are active for a given
                           task; only relevant if this function is being used in a
                           continual learning scenario
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
    :param progress_bar: Unused
    :param active_classes: Unused
    :return: mean loss for epoch
    :rtype: float
    """
    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(target)
        if transform_to_device_fn is None:
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)
        else:
            data, target = transform_to_device_fn(
                data, target, device, non_blocking=async_gpu
            )
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()
        # Output will be a list of list of tensors:
        # output[x][k] = bilinear_module_x, prediction_step_k: tensor
        output = model(data)

        # Module specific losses will be a tensor of dimension num modules
        # module_specific_losses[x] = loss from bilinear_module_x
        module_specific_losses = criterion(output, target)
        error_loss = module_specific_losses.sum()

        del data, target, output

        t2 = time.time()
        error_loss.backward()
        t3 = time.time()

        # Compute and backpropagate the complexity loss. This happens after
        # error loss has backpropagated, freeing its computation graph, so the
        # two loss functions don't compete for memory.
        complexity_loss = (
            complexity_loss_fn(model) if complexity_loss_fn is not None else None
        )
        if complexity_loss is not None:
            complexity_loss.backward()

        t4 = time.time()
        optimizer.step()
        t5 = time.time()

        if post_batch_callback is not None:
            time_string = (
                "Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                "complexity loss forward/backward: {:.3f}s," + "weight update: {:.3f}s"
            ).format(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)
            post_batch_callback(
                model=model,
                error_loss=error_loss.detach(),
                complexity_loss=(
                    complexity_loss.detach() if complexity_loss is not None else None
                ),
                batch_idx=batch_idx,
                num_images=num_images,
                time_string=time_string,
            )
        del error_loss, complexity_loss, module_specific_losses
        t0 = time.time()


def evaluate_gim_model(
    model,
    loader,
    device,
    batches_in_epoch=sys.maxsize,
    criterion=F.nll_loss,
    complexity_loss_fn=None,
    active_classes=None,
    progress=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
):
    """Evaluate pre-trained model using given test dataset loader.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: test dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device`
    :param batches_in_epoch: Max number of mini batches to test on
    :type batches_in_epoch: int
    :param criterion: loss function to use
    :type criterion: function
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param active_classes: a list of indices of the heads that are active for a given
                           task; only relevant if this function is being used in a
                           continual learning scenario
    :type active_classes: list of int or None
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
    :param progress: Unused


    :return: dictionary with computed "mean_accuracy", "mean_loss", "total_correct".
    :rtype: dict
    """
    model.eval()
    total = 0

    # Perform accumulation on device, avoid paying performance cost of .item()
    loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)

    async_gpu = loader.pin_memory
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= batches_in_epoch:
                break

            if transform_to_device_fn is None:
                data = data.to(device, non_blocking=async_gpu)
                target = target.to(device, non_blocking=async_gpu)
            else:
                data, target = transform_to_device_fn(
                    data, target, device, non_blocking=async_gpu
                )

            output = model(data)
            if active_classes is not None:
                output = output[:, active_classes]
            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

            if post_batch_callback is not None:
                post_batch_callback(
                    batch_idx=batch_idx, target=target, output=output, pred=pred
                )

        complexity_loss = (
            complexity_loss_fn(model) if complexity_loss_fn is not None else None
        )
    correct = correct.item()
    loss = loss.item()

    result = {
        "total_correct": correct,
        "total_tested": total,
        "mean_loss": loss / total if total > 0 else 0,
        "mean_accuracy": correct / total if total > 0 else 0,
    }

    if complexity_loss is not None:
        result["complexity_loss"] = complexity_loss.item()

    return result
