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


import copy

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from nupic.research.frameworks.pytorch import datasets
from nupic.research.frameworks.vernon import interfaces

__all__ = [
    "LegacyImagenetConfig",
]


class LegacyImagenetConfig(
    interfaces.Experiment,  # Requires
):
    """
    Converts the SupervisedExperiment into the ImagenetExperiment that many
    experiments are configured to use.

    The following arguments are added to the base experiment config.

        - batch_norm_weight_decay: Whether or not to apply weight decay to
                                   batch norm modules parameters
                                   See https://arxiv.org/abs/1807.11205
        - bias_weight_decay: Whether or not to apply weight decay to
                                   bias parameters
    """

    @classmethod
    def load_dataset(cls, config, train=True):
        config = copy.copy(config)
        config.setdefault("dataset_class", datasets.imagenet)
        if "dataset_args" not in config:
            config["dataset_args"] = dict(
                data_path=config["data"],
                train_dir=config.get("train_dir", "train"),
                val_dir=config.get("val_dir", "val"),
                num_classes=config.get("num_classes", 1000),
                use_auto_augment=config.get("use_auto_augment", False),
                sample_transform=config.get("sample_transform", None),
                target_transform=config.get("target_transform", None),
                replicas_per_sample=config.get("replicas_per_sample", 1),
            )

        return super().load_dataset(config, train)

    @classmethod
    def should_decay_parameter(cls, module, parameter_name, parameter, config):
        if isinstance(module, _BatchNorm):
            return config.get("batch_norm_weight_decay", True)
        elif parameter_name == "bias":
            return config.get("bias_weight_decay", True)
        else:
            return True

    @classmethod
    def create_optimizer(cls, config, model):
        # Configure optimizer
        group_decay, group_no_decay = [], []
        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if cls.should_decay_parameter(module, name, param, config):
                    group_decay.append(param)
                else:
                    group_no_decay.append(param)

        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        optimizer = optimizer_class([dict(params=group_decay),
                                     dict(params=group_no_decay,
                                          weight_decay=0.)],
                                    **optimizer_args)
        return optimizer

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["load_dataset"].insert(0, "ImagenetExperiment: Set default dataset")
        return eo
