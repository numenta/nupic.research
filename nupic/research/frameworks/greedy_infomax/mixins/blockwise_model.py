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
from nupic.research.frameworks.vernon.network_utils import create_model
import torch


class BlockWiseModel:

    """
    This method recursively instantiates models, allowing users to use hierarchical
    model configs and pass nn.Module instances as arguments. This is used whenever
    you might need to load modules independently from different checkpoint files.

    Example model config:

    model_class = SomeSequentialModel
    model_args = dict(
            modules=[
                dict(
                    model_class=SparseWeights2D,
                    model_args=dict(
                            instantia
                        )
                ),
                dict(
                    model_class=nn.Conv2d,
                    model_args=dict(
                            in_channels=10,
                            out_channels=20
                        )
                ),
            ]
        other_arg=True
        another_arg=1.0
    )

    """
    @classmethod
    def create_model(cls, config, device):
        model_args = config.get("model_args", {})
        for k, v in model_args.items():
            if isinstance(v, dict) and "model_class" in v:
                # this is a model that needs to be created




        # if "instantiate_modules" in model_args.keys():
        #     instantiated_modules = []
        #     for module_dict in model_args["instantiate_modules"]:
        #         module = create_model(
        #             model_class=module_dict["model_class"],
        #             model_args=module_dict.get("model_args", {}),
        #             init_batch_norm=module_dict.get("init_batch_norm", False),
        #             device=device,
        #             checkpoint_file=module_dict.get("checkpoint_file", None),
        #             load_checkpoint_args=module_dict.get("load_checkpoint_args", {}),
        #         )
        #     model_args.update(model_blocks=model_blocks)
        return create_model(
            model_class=config["model_class"],
            model_args=model_args,
            init_batch_norm=config.get("init_batch_norm", False),
            device=device,
            checkpoint_file=config.get("checkpoint_file", None),
            load_checkpoint_args=config.get("load_checkpoint_args", {}),
        )

    @classmethod
    def create_optimizer(cls, config, model):
        """
        Create optimizer from an experiment config.

        :param optimizer_class: Callable or class to instantiate optimizer. Must return
                                object inherited from "torch.optim.Optimizer"
        :param optimizer_args: Arguments to pass to the optimizer.
        """
        if "model_blocks" in config.keys():
            parameters_to_train = []
            for module, module_args in zip(model.modules, config["model_blocks"]):
                if module_args["train"]:
                    parameters_to_train.append(module.parameters())
        else:
            parameters_to_train = model.parameters()
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class(parameters_to_train, **optimizer_args)