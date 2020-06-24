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


class CutMix(object):
    """
    Applies CutMix (Mixup + CutOut) regularization approach
    Paper: https://arxiv.org/pdf/1905.04899.pdf

    Cutmix is considered to be a state of the art algorithm in regularization.
    It combines two ideas:
    - Mixup: mixes two classes by mixing both the input and the target.
    Mixup linearly combines the inputs, and can distort the image to unrealistic
    settings.
    - Cutout: Also known as regional dropout, replaces patches of the image
    by either black patches or patches with random noise.
    Replacing a patch of the image incurs in loss of information.

    Cutmix combine both methods, by replacing random patches in the image
    with patches from other images in the dataset,
    and combining the target labels accordingly.
    """
    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - mixup_beta: Parameters for the beta distribution used in mixup to draw
                          lambda, which defines the size of the bounding boxes.
                          The combination ratio λ between two data points is sampled
                          from the distribution Beta(mixup_beta, mixup_beta).
                          If set to 1, λ is sampled from the uniform distribution.
            - cutmix_prob: Probability to apply cutmix at each batch.
        """
        super().setup_experiment(config)

        # CutMix variables: fixate beta for now
        self.mixup_beta = config.get("mixup_beta", 1.0)
        self.cutmix_prob = config.get("cutmix_prob", 1.0)

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
        target = target.to(self.device, non_blocking=non_blocking)

        # transform the data - generate mixed sample
        lam = np.random.beta(self.mixup_beta, self.mixup_beta)
        rand_index = torch.randperm(data.shape[0], device=self.device)
        # draw and apply the bounding boxes to batch
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.shape[-1] * data.shape[-2]))
        # combine the targets, will require one hot
        ohe_target = F.one_hot(target, num_classes=self.num_classes)
        ohe_target_patches = F.one_hot(target[rand_index], num_classes=self.num_classes)
        new_target = lam * ohe_target + (1. - lam) * ohe_target_patches

        return data, new_target

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
        eo["setup_experiment"].append("Cutmix parameters")
        eo["transform_data_to_device"].insert(0, "If not training: {")
        eo["transform_data_to_device"].append(
            "} else: { Compute Cutmix targets }"
        )
        eo["error_loss"].insert(0, "If not training: {")
        eo["error_loss"].append(
            "} else: { Soft cross entropy loss }"
        )
        return eo


class CutMixKnowledgeDistillation(CutMix):
    """
    Applies CutMix (Mixup + CutOut) regularization approach
    Paper: https://arxiv.org/pdf/1905.04899.pdf
    """
    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - mixup_beta: Parameters for the beta distribution used in mixup to draw
                          lambda, which defines the size of the bounding boxes.
                          The combination ratio λ between two data points is sampled
                          from the distribution Beta(mixup_beta, mixup_beta).
                          If set to 1, λ is sampled from the uniform distribution.
            - cutmix_prob: Probability to apply cutmix at each batch.
            - teacher_model_class: Class for pretrained model to be used as teacher
                                   in knowledge distillation.
        """
        super().setup_experiment(config)

        # CutMix variables
        self.mixup_beta = config.get("mixup_beta", 1.0)
        self.cutmix_prob = config.get("cutmix_prob", 1.0)

        # Teacher model and knowledge distillation variables
        teacher_model_class = config.get("teacher_model_class", None)
        assert teacher_model_class is not None, \
            "teacher_model_class must be specified for KD experiments"
        self.teacher_model = teacher_model_class()
        self.teacher_model.eval()
        self.teacher_model.to(self.device)
        self.logger.info(f"KD teacher class: {teacher_model_class}")

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

        # transform the data - generate mixed sample
        lam = np.random.beta(self.mixup_beta, self.mixup_beta)
        rand_index = torch.randperm(data.shape[0], device=self.device)
        # draw and apply the bounding boxes to batch
        bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
        data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.shape[-1] * data.shape[-2]))

        # recalculate softmax target
        with torch.no_grad():
            soft_target = F.softmax(self.teacher_model(data))

        # combine the targets, will require one hot
        soft_target_patches = F.one_hot(target[rand_index],
                                        num_classes=self.num_classes)
        new_target = lam * soft_target + (1. - lam) * soft_target_patches

        # no extra memory - just regular target, a vector with indexes and a scalar
        return data, new_target

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Cutmix and Knowledge Distillation parameters")
        eo["transform_data_to_device"].insert(0, "If not training: {")
        eo["transform_data_to_device"].append(
            "} else: { Compute Cutmix targets using soft target from teacher }"
        )
        eo["error_loss"].insert(0, "If not training: {")
        eo["error_loss"].append(
            "} else: { Mixup composite loss }"
        )
        return eo


def rand_bbox(shape, lam):
    """
    Defines random bounding boxes around the image
    Lambda factor defines the size of the bounding box
    from: https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    """

    width = shape[2]
    height = shape[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(width * cut_rat)
    cut_h = np.int(height * cut_rat)

    # uniform
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


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
