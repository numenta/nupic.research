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

import torch
import torch.nn.functional as F


class KnowledgeDistillation(object):
    """
    Sets the network to learn from a teacher model
    """
    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - teacher_model_class: Class for pretrained model to be used as teacher
                                   in knowledge distillation.
            - kd_factor_init: Determines the percentage of the target that comes
                              from the teacher model. Value should be float
                              between 0 and 1. Defaults to 1.
            - kd_factor_end: KD factor at last epoch. Will calculate linear decay
                             based on initial kd_factor_init and kd_factor_end.
                             Value should be float between 0 and 1.
                             If None, no decay is applied. Defaults to None.
        """
        super().setup_experiment(config)

        # Teacher model and knowledge distillation variables
        teacher_model_class = config.get("teacher_model_class", None)
        assert teacher_model_class is not None, \
            "teacher_model_class must be specified for KD experiments"
        self.teacher_model = teacher_model_class()

        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        self.logger.info(f"KD teacher class: {teacher_model_class}")

        # initalize Knowledge Distillation factor
        self.kd_factor_init = config.get("kd_factor_init", 1)
        assert 0 <= self.kd_factor_init <= 1, \
            "kd_factor_init should be >= 0 and <= 1"
        self.kd_factor_end = config.get("kd_factor_end", None)
        if self.kd_factor_end is not None:
            assert 0 <= self.kd_factor_end <= 1, \
                "kd_factor_end should be >= 0 and <= 1"
        else:
            self.kd_factor = self.kd_factor_init
        self.logger.info(f"KD factor: {self.kd_factor_init} {self.kd_factor_end}")

    def pre_epoch(self):
        super().pre_epoch()
        # calculates kd factor based on a linear decay
        if self.kd_factor_end is not None:
            self.kd_factor = linear_decay(first_epoch_value=self.kd_factor_init,
                                          last_epoch_value=self.kd_factor_end,
                                          current_epoch=self.current_epoch,
                                          total_epochs=self.epochs)
            self.logger.debug(
                f"KD factor: {self.kd_factor:.3f} at epoch {self.current_epoch}")

    def send_data_to_device(self, data, target, device, non_blocking):
        """
        :param data: input to the training function, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param device: identical to self.device
        :param non_blocking: define whether or not to use
                             asynchronous GPU copies when the memory is pinned
        """
        if not self.model.training:
            return super().send_data_to_device(data, target, device,
                                               non_blocking)

        data = data.to(self.device, non_blocking=non_blocking)
        with torch.no_grad():
            # target is linear combination of teacher and target softmaxes
            softmax_output_teacher = F.softmax(self.teacher_model(data))
            if self.kd_factor < 1:
                target = target.to(self.device, non_blocking=non_blocking)
                one_hot_target = F.one_hot(target, num_classes=self.num_classes)
                combined_target = (self.kd_factor * softmax_output_teacher
                                   + (1 - self.kd_factor) * one_hot_target)
                del target, one_hot_target
            else:
                combined_target = softmax_output_teacher

        return data, combined_target

    def error_loss(self, output, target, reduction="mean"):
        """
        :param output: output from the model
        :param target: target to be matched by model
        :param reduction: reduction to apply to the output ("sum" or "mean")
        """
        if not self.model.training:
            # Targets are from the dataloader
            return super().error_loss(output, target, reduction=reduction)

        return soft_cross_entropy(output, target, reduction)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Knowledge Distillation initialization")
        eo["pre_epoch"].append("Update kd factor based on linear decay")
        eo["send_data_to_device"].insert(0, "If not training: {")
        eo["send_data_to_device"].append(
            "} else: { Compute Knowledge Distillation targets }"
        )
        eo["error_loss"].insert(0, "If not training: {")
        eo["error_loss"].append(
            "} else: { Knowledge Distillation soft_cross_entropy }"
        )
        return eo


def soft_cross_entropy(output, target, reduction="mean"):
    """ Cross entropy that accepts soft targets
    Args:
    :param output: predictions for neural network
    :param targets: targets, can be soft
    :param size_average: if false, sum is returned instead of mean

    Examples::

        output = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        output = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(output, target)
        loss.backward()

    see: https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/5
    """
    if reduction == "mean":
        return torch.mean(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))
    elif reduction == "sum":
        return torch.sum(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def linear_decay(first_epoch_value, last_epoch_value, current_epoch, total_epochs):
    """
    Calculates value for a current epoch in a linear decay.

    :param first_epoch_value: Value at first epoch (before training).
    :param last_epoch_value: Value at last epoch (before training).
    :param current_epoch: Current epoch. Assumes first epoch is 0.
    :param total_epochs: Total number of epochs in training.
    """
    step_size = (first_epoch_value - last_epoch_value) / (total_epochs - 1)
    return first_epoch_value - step_size * current_epoch
