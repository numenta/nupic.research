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
from tqdm import tqdm
from .loss_utils import multiple_cross_entropy

def train_block_model(
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
    :type active_classes: list of int or None
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
        if transform_to_device_fn is None:
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)
        else:
            data, target = transform_to_device_fn(data, target, device,
                                                  non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()
        # Output will be a list of list of tensors:
        # output[x][k] = bilinear_module_x, prediction_step_k: tensor
        output_list = model(data)

        # Module specific losses will be a tensor of dimension num modules
        # module_specific_losses[x] = loss from bilinear_module_x
        module_losses = criterion(output_list, target)
        error_loss = module_losses.sum()

        del data, target, output_list

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
                                time_string=time_string,
                                module_losses=module_losses.detach())
        del error_loss, complexity_loss, module_losses
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


def evaluate_block_model(
    model,
    loader,
    device,
    batches_in_epoch=sys.maxsize,
    criterion=multiple_cross_entropy,
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
    :param progress: Optional :class:`tqdm` progress bar args. None for no progress bar
    :type progress: dict or None
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
    total = 0

    # Perform accumulation on device, avoid paying performance cost of .item()
    num_emit_encoding_modules = model.encoder.count_emit_encoding_modules()
    module_losses = torch.zeros(num_emit_encoding_modules, device=device)
    module_correct = torch.zeros(num_emit_encoding_modules, device=device)

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

            outputs = model(data)
            if active_classes is not None:
                outputs = outputs[:, :, active_classes] #module, batch, classes
            module_losses += criterion(outputs, target, reduction="sum")
            preds = outputs.max(-1, keepdim=True)[1]
            module_correct += preds.eq(target.view_as(preds)).sum((1, 2))
            total += len(data)

            if post_batch_callback is not None:
                post_batch_callback(batch_idx=batch_idx, target=target, output=outputs,
                                    preds=preds)

        complexity_loss = (complexity_loss_fn(model)
                           if complexity_loss_fn is not None
                           else None)

    if progress is not None:
        loader.close()

    result = {
        "total_tested": total,
        "total_correct": module_correct[-1].item(),
        "mean_loss": module_losses[-1].item() / total,
        "mean_accuracy": module_correct[-1].item() / total,
    }
    result.update({
        "num_bilinear_info_modules" : model.encoder.count_bilinear_info_modules(),
        "num_emit_encodings": model.encoder.count_emit_encoding_modules(),
    })
    result.update({
        f"total_correct_encoding_{i}": module_correct[i].item() for i in range(
            module_correct.shape[0])
    })
    result.update({
        f"mean_loss_encoding_{i}": module_losses[i].item() / total if total > 0
        else 0 for i in range(module_correct.shape[0])
    })
    result.update({
        f"mean_accuracy_encoding_{i}": module_correct[i].item() / total if total > 0
        else 0
        for i in range(module_correct.shape[0])
    })

    if complexity_loss is not None:
        result["complexity_loss"] = complexity_loss.item()

    return result
