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
                                   in knowledge distillation. Can be one or list of
                                   classes for knowledge distillation with ensemble.
            - kd_ensemble_weights: List of weights to apply to each teacher model
                                   during distillation.
                                   If the total is > 1 the loss will be scaled out
                                   of proportion, acting in practice as a scaling factor
                                   to the learning rate (the equivalence is true
                                   in the composite loss model, and only approximate
                                   for the regular distillation model. Scaling the
                                   softmax out of proportion creates a target that
                                   is impossible to reach, since the output distribution
                                   can only sum to 1)
            - kd_factor_init: Determines the percentage of the target that comes
                              from the teacher model. Value should be float
                              between 0 and 1. Defaults to 1.
            - kd_factor_end: KD factor at last epoch. Will calculate linear decay
                             based on initial kd_factor_init and kd_factor_end.
                             Value should be float between 0 and 1.
                             If None, no decay is applied. Defaults to None.
            - kd_temperature_init: Determines the temperature T applied to softmax.
                                   If T > 1, it smoothes the softmax distribution.
                                   If T < 1, it sharpens the distribution (more mass to
                                   few points). If kd_temperature_end is also defined,
                                   this variable equals the temperature at the beginning
                                   of training. Defaults to 1.0
            - kd_temperature_end: Determines the temperature applied to softmax.
                                  Will calculate linear decay based on
                                  kd_temperature_init and kd_temperature_end.
                                  If None, no decay is applied. Defaults to None.
        """
        super().setup_experiment(config)

        # Teacher model and knowledge distillation variables
        teacher_model_class = config.get("teacher_model_class", None)
        assert teacher_model_class is not None, \
            "teacher_model_class must be specified for KD experiments"

        # convert into a list of teachers
        if type(teacher_model_class) != list:
            self.logger.info(f"KD single teacher class: {teacher_model_class}")
            teacher_model_class = [teacher_model_class]

        self.teacher_models = [
            model().eval().to(self.device) for model in teacher_model_class
        ]
        if len(self.teacher_models) > 1:
            self.logger.info(f"KD teacher is ensemble of "
                             f"{len(self.teacher_models)} models")

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

        # initalize Knowledge softmax temperature factor
        self.kd_temperature_init = config.get("kd_temperature_init", 1.0)
        self.kd_temperature_end = config.get("kd_temperature_end", None)
        if self.kd_temperature_end is None:
            self.kd_temperature = self.kd_temperature_init
        self.logger.info("KD softmax temperature: "
                         f"{self.kd_temperature_init} {self.kd_temperature_end}")

        # initialize ensemble weighting
        self.kd_ensemble_weights = config.get("kd_ensemble_weights", None)
        if self.kd_ensemble_weights is None:
            num_models = len(self.teacher_models)
            self.kd_ensemble_weights = [1. / num_models for _ in range(num_models)]
        else:
            assert len(self.kd_ensemble_weights) == len(self.teacher_models), \
                "Number of ensemble weights should match number of teacher models"
        self.logger.info(f"Ensemble weights: {self.kd_ensemble_weights}")

    def pre_epoch(self):
        super().pre_epoch()

        # calculates kd factor based on a linear decay
        if self.kd_factor_end is not None:
            self.kd_factor = linear_decay(first_epoch_value=self.kd_factor_init,
                                          last_epoch_value=self.kd_factor_end,
                                          current_epoch=self.current_epoch,
                                          total_epochs=self.epochs)
            self.logger.debug(
                f"KD factor: {self.kd_factor:.3f} @ epoch {self.current_epoch}")

        # calculates softmax temperature based on a linear decay
        if self.kd_temperature_end is not None:
            self.kd_temperature = linear_decay(
                first_epoch_value=self.kd_temperature_init,
                last_epoch_value=self.kd_temperature_end,
                current_epoch=self.current_epoch,
                total_epochs=self.epochs
            )
            self.logger.debug(
                f"KD temperature: {self.kd_temperature:.3f} @ epoch{self.current_epoch}"
            )

    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        :param data: input to the training function, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param device: identical to self.device
        :param non_blocking: define whether or not to use
                             asynchronous GPU copies when the memory is pinned
        """
        if not self.model.training:
            return super().transform_data_to_device(data, target, device,
                                                    non_blocking)

        data = data.to(self.device, non_blocking=non_blocking)
        with torch.no_grad():
            # if ensemble, linearly combine outputs of softmax
            softmax_output_teacher = None
            for wfactor, tmodel in zip(self.kd_ensemble_weights, self.teacher_models):
                if softmax_output_teacher is None:
                    softmax_output_teacher = \
                        F.softmax(tmodel(data) / self.kd_temperature) * wfactor
                else:
                    softmax_output_teacher += \
                        F.softmax(tmodel(data) / self.kd_temperature) * wfactor

            if self.kd_factor < 1:
                # target is linear combination of teacher and target softmaxes
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
        eo["transform_data_to_device"].insert(0, "If not training: {")
        eo["transform_data_to_device"].append(
            "} else: { Compute Knowledge Distillation targets }"
        )
        eo["error_loss"].insert(0, "If not training: {")
        eo["error_loss"].append(
            "} else: { Knowledge Distillation soft_cross_entropy }"
        )
        return eo


class KnowledgeDistillationCL(KnowledgeDistillation):
    """
    Alternative version of knowledge distillation that combines the loss functions
    instead of linearly combining the softmax outputs
    """

    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        :param data: input to the training function, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param device: identical to self.device
        :param non_blocking: define whether or not to use
                             asynchronous GPU copies when the memory is pinned
        """
        if not self.model.training:
            return super().transform_data_to_device(data, target, device,
                                                    non_blocking)

        target = target.to(self.device, non_blocking=non_blocking)
        data = data.to(self.device, non_blocking=non_blocking)

        # calculate and return soft targets for each model
        with torch.no_grad():
            soft_targets = []
            for tmodel in self.teacher_models:
                soft_targets.append(F.softmax(tmodel(data) / self.kd_temperature))

        return data, (target, soft_targets)

    def error_loss(self, output, target, reduction="mean"):
        """
        :param output: output from the model
        :param target: target to be matched by model
        :param reduction: reduction to apply to the output ("sum" or "mean")
        """
        if not self.model.training:
            # Targets are from the dataloader
            return super().error_loss(output, target, reduction=reduction)

        # unpack targets
        real_target, soft_targets = target

        # combine several models
        kd_error_loss = 0
        for wfactor, soft_target in zip(self.kd_ensemble_weights, soft_targets):
            kd_error_loss += soft_cross_entropy(output, soft_target) * wfactor
        del soft_targets

        # combine with regular target if kd_factor < 1
        if self.kd_factor < 1:
            true_error_loss = F.cross_entropy(output, target)
            error_loss = (self.kd_factor * kd_error_loss
                          + (1 - self.kd_factor) * true_error_loss)
        else:
            error_loss = kd_error_loss
        del target, output

        return error_loss


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
