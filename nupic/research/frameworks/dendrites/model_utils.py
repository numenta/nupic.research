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

import sys

import torch
import torch.nn.functional as F

__all__ = [
    "evaluate_dendrite_model",
    "train_dendrite_model",
]


def train_dendrite_model(
    model,
    loader,
    optimizer,
    device,
    criterion=F.cross_entropy,
    share_labels=False,
    num_labels=None,
    active_classes=None,
    train_context_fn=None,
    context_vector=None,
    post_batch_callback=None,
    complexity_loss_fn=None,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    transform_to_device_fn=None,
    progress_bar=None,
):
    """
    Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set. This function is compatible with models
    and datasets that explicitly provide context, and ones that don't.

    For unused parameters, see `nupic.research.frameworks.pytorch.model_utils`.

    :param model: Pytorch model to be trained
    :param loader: Train dataset loader
    :param optimizer: Optimizer object used to train the model. This function
                      will train the model on every batch using this optimizer and the
                      :func:`torch.nn.functional.nll_loss` function
    :param device: Device to use ('cpu' or 'cuda')
    :param criterion: Loss function to use
    :param share_labels: Whether or not to share labels between tasks, which is
                         accomplished by applying the modulo operator to target labels
    :param num_labels: The number of unique labels; only required if
                       `share_labels = True`
    :param active_classes: List of indices of the heads that are active for a given
                           task; only relevant if this function is being used in a
                           continual learning scenario
    :param context_vector: If not None, use this context vector in place of any that
                           we get from the loader
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :param complexity_loss_fn: Unused
    :param batches_in_epoch: Unused
    :param pre_batch_callback: Unused
    :param transform_to_device_fn: Unused
    :param progress_bar: Unused
    """
    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    context = None
    if context_vector is not None:
        # Tile context vector
        context = context_vector.repeat(loader.batch_size, 1)

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        # `data` may be a 2-item list comprising the example data and context signal in
        # case context is explicitly provided
        if isinstance(data, list):
            if context_vector is None:
                data, context = data
            else:
                data, _ = data
        data = data.to(device, non_blocking=async_gpu)
        data = data.flatten(start_dim=1)
        target = target.to(device, non_blocking=async_gpu)

        if train_context_fn is not None:
            context = train_context_fn(data)

        # Since labels are shared, target values should be in
        # [0, 1, ..., num_labels - 1]
        if share_labels:
            target = target % num_labels

        if context is not None:
            context = context.to(device, non_blocking=async_gpu)

        # FIXME: Pytorch 1.7: Replace with optimizer.zero_grad(set_to_none=True)
        # optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            for p in group["params"]:
                p.grad = None

        forward_args = [data] if context is None else [data, context]
        output = model(*forward_args)
        if active_classes is not None:
            output = output[:, active_classes]

        error_loss = criterion(output, target)
        error_loss.backward()
        optimizer.step()

        # Rezero weights if necessary
        if post_batch_callback is not None:
            post_batch_callback(model=model, error_loss=error_loss.detach(),
                                complexity_loss=None, batch_idx=batch_idx,
                                num_images=0, time_string="")


def evaluate_dendrite_model(
    model,
    loader,
    device,
    criterion=F.cross_entropy,
    share_labels=False,
    num_labels=None,
    active_classes=None,
    infer_context_fn=None,
    batches_in_epoch=None,
    complexity_loss_fn=None,
    progress=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
):
    """
    Evaluate pre-trained model using given test dataset loader, and return a dict with
    computed "mean_accuracy", "mean_loss", "total_correct", and "total_tested". This
    function is compatible with models and datasets that explicitly provide context,
    and ones that don't.

    For unused parameters, see `nupic.research.frameworks.pytorch.model_utils`.

    :param model: Pretrained pytorch model
    :param loader: Test dataset loader
    :param device: Device to use ('cpu' or 'cuda')
    :param criterion: Loss function to use
    :param share_labels: Whether or not to share labels between tasks, which is
                         accomplished by applying the modulo operator to target labels
    :param num_labels: The number of unique labels; only required if
                       `share_labels = True`
    :param active_classes: List of indices of the heads that are active for a given
                           task; only relevant if this function is being used in a
                           continual learning scenario
    :infer_context_fn: A function that computes the context vector to use given a batch
                       of data samples
    :param batches_in_epoch: Unused
    :param complexity_loss_fn: Unused
    :param progress: Unused
    :param post_batch_callback: Unused
    :param transform_to_device_fn: Unused
    """
    model.eval()
    total = 0

    loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)
    context = None

    with torch.no_grad():

        for data, target in loader:

            # `data` may be a 2-item list comprising the example data and context
            # signal in case context is explicitly provided
            if isinstance(data, list):
                data, context = data
            data = data.flatten(start_dim=1)

            # Since labels are shared, target values should be in
            # [0, 1, ..., num_labels - 1]
            if share_labels:
                target = target % num_labels

            data = data.to(device)
            target = target.to(device)
            if infer_context_fn is not None:
                # Use `infer_context_fn` to retrieve the context vector
                context = infer_context_fn(data)
            if context is not None:
                context = context.to(device)

            forward_args = [data] if context is None else [data, context]
            output = model(*forward_args)
            if active_classes is not None:
                output = output[:, active_classes]

            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    results = {
        "total_correct": correct.item(),
        "total_tested": total,
        "mean_loss": loss.item() / total if total > 0 else 0,
        "mean_accuracy": torch.true_divide(correct, total).item() if total > 0 else 0,
    }
    return results
