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
    """
    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        :param data: input to the model, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param device: identical to self.device
        :param non_blocking: define whether or not to use
                             asynchronous GPU copies when the memory is pinned
        """
        if not self.model.training:
            return super().transform_data_to_device(data, target, device,
                                                    non_blocking)

        if len(data.shape) < 5:
            raise ValueError("Define replicas_per_sample > 1")

        data = data.to(self.device, non_blocking=non_blocking)
        target = target.to(self.device, non_blocking=non_blocking)

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

        return data_variant, target

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["transform_data_to_device"].insert(0, "If not training: {")
        eo["transform_data_to_device"].append(
            "} else: { MaxupStandard: Choose data variant }"
        )
        return eo


class MaxupPerSample(object):
    """
    Maxup: applies m transformations to each sample;
    composes a batch with the transformation of each sample
    with highest loss. Equivalent to formal description
    in the methods section of the paper.

    Paper: https://arxiv.org/pdf/2002.09024.pdf
    """
    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        :param data: input to the model, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param device: identical to self.device
        :param non_blocking: define whether or not to use
                             asynchronous GPU copies when the memory is pinned
        """
        if not self.model.training:
            return super().transform_data_to_device(data, target, device,
                                                    non_blocking)

        # calculate loss for all the tranformed versions of the image
        if len(data.shape) < 5:
            raise ValueError("Define replicas_per_sample > 1")

        data = data.to(self.device, non_blocking=non_blocking)
        target = target.to(self.device, non_blocking=non_blocking)

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

        return data_variant, target

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["transform_data_to_device"].insert(0, "If not training: {")
        eo["transform_data_to_device"].append(
            "} else: { MaxupPerSample: Choose data variant }"
        )
        return eo


def sample_cross_entropy(output, target):
    """ Cross entropy of a single sample. Accepts soft targets
    :param output: predictions for neural network
    :param targets: targets, can be soft
    :return: cross entropy per sample, not aggregated
    """
    return torch.sum(-target * F.log_softmax(output), dim=1)
