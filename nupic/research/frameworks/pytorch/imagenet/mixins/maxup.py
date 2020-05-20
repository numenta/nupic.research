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

import numpy as np
import torch
import torch.nn.functional as F


class MaxupStandard(object):
    """
    Maxup: applies m transformations to each sample;
    concatenates into m different batches;
    learns from batch with worst loss.

    Paper: https://arxiv.org/pdf/2002.09024.pdf

    Requires ComplexLoss Mixin
    """
    def __init__(self):
        super().__init__()

        self.execution_order["calculate_batch_loss"] = [
            "MaxupStandard.calculate_batch_loss"
        ]

    def calculate_batch_loss(self, data, target, async_gpu=True):
        """
        :param data: input to the training function, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param async_gpu: define whether or not to use
                          asynchronous GPU copies when the memory is pinned
        """

        if len(data.shape) < 5:
            raise ValueError("Define replicas_per_sample > 1")

        target = target.to(self.device, non_blocking=async_gpu)

        # calculate loss for all the different variants of the batch
        losses = []
        replicas_per_sample = data.shape[1]
        for dim in range(replicas_per_sample):
            data_variant = data[:, dim, :, :, :]
            with torch.no_grad():
                output = self.model(data_variant)
                losses.append(self.loss_function(output, target).item())

        # choose the max loss
        max_loss_dim = np.argmax(losses)

        # regular training with the max loss
        data_variant = data[:, max_loss_dim, :, :, :]
        output = self.model(data_variant)
        loss = self.loss_function(output, target)

        del data, data_variant, target, losses
        return loss, output


class MaxupPerSample(object):
    """
    Maxup: applies m transformations to each sample;
    composes a batch with the transformation of each sample
    with highest loss. Equivalent to formal description
    in the methods section of the paper.

    Paper: https://arxiv.org/pdf/2002.09024.pdf

    Requires ComplexLoss Mixin
    """
    def __init__(self):
        super().__init__()

        self.execution_order["calculate_batch_loss"] = [
            "MaxupPerSample.calculate_batch_loss"
        ]

    def calculate_batch_loss(self, data, target, async_gpu=True):
        """
        :param data: input to the training function, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param async_gpu: define whether or not to use
                          asynchronous GPU copies when the memory is pinned
        """
        # calculate loss for all the tranformed versions of the image
        if len(data.shape) < 5:
            raise ValueError("Define replicas_per_sample > 1")

        target = target.to(self.device, non_blocking=async_gpu)

        # calculate loss for all the tranformed versions of the image
        losses = []
        one_hot_target = None
        replicas_per_sample = data.shape[1]
        for dim in range(replicas_per_sample):
            data_variant = data[:, dim, :, :, :]

            # calculate cross entropy per sample
            with torch.no_grad():
                output = self.model(data_variant)
                if one_hot_target is None:
                    one_hot_target = F.one_hot(target, num_classes=output.shape[-1])
                losses.append(sample_cross_entropy(output, one_hot_target))

        # 0-dim is number of maxup images, 1-dim is batch size
        losses = torch.stack(losses)
        # get max over the 0-dim, number of replicas per sample
        max_indices = torch.argmax(losses, dim=0)
        # use max indices to select locally
        data_variant = data[range(len(target)), max_indices, :, :, :]
        output = self.model(data_variant)
        loss = self.loss_function(output, target)

        del data, data_variant, target, one_hot_target, losses, max_indices
        return loss, output


def sample_cross_entropy(output, target):
    """ Cross entropy of a single sample. Accepts soft targets
    :param output: predictions for neural network
    :param targets: targets, can be soft
    :return: cross entropy per sample, not aggregated
    """
    return torch.sum(-target * F.log_softmax(output), dim=1)
