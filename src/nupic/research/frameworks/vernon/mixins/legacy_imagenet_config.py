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
    def create_optimizer(cls, config, model):
        """
        Modify config to work with `ConfigureOptimizerParamGroups` mixin.
        """
        no_decay_params = dict()

        # Turn off weight decay for batch_norm modules.
        if not config.get("batch_norm_weight_decay", True):
            no_decay_params.update(
                include_modules=[_BatchNorm]
            )

        # Turn off weight decay for bias parameters.
        if not config.get("bias_weight_decay", True):
            no_decay_params.update(
                include_patterns=[".*bias"]
            )

        # Add `optim_args_groups` to configure no weight decay on batchnorm and bias
        # parameters.
        if no_decay_params:
            config.update(
                optim_args_groups=[
                    dict(
                        group_args=dict(
                            weight_decay=0,
                        ),
                        **no_decay_params
                    )
                ]
            )

        return super().create_optimizer(config, model)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "ImagenetExperiment: "
        eo["load_dataset"].insert(0, name + "Set default dataset")
        eo["create_optimizer"].insert(0, name + "Modify config to work with "
                                                "ConfigureOptimizerParamGroups mixin.")
        return eo
