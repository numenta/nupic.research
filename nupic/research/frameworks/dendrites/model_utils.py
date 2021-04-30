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
    post_batch_callback=None,
    complexity_loss_fn=None,
    batches_in_epoch=None,
    active_classes=None,
    pre_batch_callback=None,
    transform_to_device_fn=None,
    progress_bar=None,
):
    """
    TODO: add docstring
    """
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        # TODO: need to make this more generic to not require context
        data, context = data
        data = data.flatten(start_dim=1)

        # Since there's only one output head, target values should be modified to be in
        # the range [0, 1, ..., 9]
        if share_labels:
            target = target % num_labels

        data = data.to(device)
        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data, context)

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
    criterion=F.nll_loss,
    share_labels=False,
    num_labels=None,
    batches_in_epoch=None,
    complexity_loss_fn=None,
    active_classes=None,
    progress=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
):
    """
    TODO docstring
    """
    model.eval()
    total = 0

    loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)

    with torch.no_grad():

        for data, target in loader:
            # TODO: need to make this more generic to not require context
            data, context = data
            data = data.flatten(start_dim=1)

            # Since there's only one output head, target values should be modified to
            # be in the range [0, 1, ..., 9]
            if share_labels:
                target = target % num_labels

            data = data.to(device)
            context = context.to(device)
            target = target.to(device)

            output = model(data, context)

            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

    results = {
        "total_correct": correct,
        "total_tested": total,
        "mean_loss": loss / total if total > 0 else 0,
        "mean_accuracy": torch.true_divide(correct, total).item() if total > 0 else 0,
    }
    return results
