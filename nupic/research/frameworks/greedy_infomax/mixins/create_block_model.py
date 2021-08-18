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

class CreateBlockModel:

    """
    This method recursively instantiates models, allowing users to use hierarchical
    model configs and pass nn.Module instances as arguments. This is used whenever
    you might need to load modules independently from different checkpoint files.

    Example model config:

    model_class = BlockGIMEncoder
    model_args = dict(
            modules=[
                dict(module=nn.Conv2d(in_features=x, out_features=y),
                train=True,
                load_checkpoint="checkpoint.file"
            ]
        other_arg=True
        another_arg=1.0
    )

    """
    @classmethod
    def create_model(cls, config, device):
        model_args = config.get("model_args", {})
        modules_to_create = model_args.get("modules", [])
        for i, module_dict in enumerate(modules_to_create):
            modules_to_create[i] = create_model(module_dict, device)
        model_args["module_instances"] = modules_to_create
        return create_model(
            model_class=config["model_class"],
            model_args=model_args,
            init_batch_norm=config.get("init_batch_norm", False),
            device=device,
            checkpoint_file=config.get("checkpoint_file", None),
            load_checkpoint_args=config.get("load_checkpoint_args", {}),
        )

    @classmethod
    def create_optimizer(cls, config, device):
        model_args = config.get("model_args", {})
        modules = model_args.get("modules", [])
        module_instances = model_args["model_instances"]
        parameters_to_train = set()
        for module_dict, module_instance in zip(modules, module_instances):
            if module_dict["train"]:
                parameters_to_train.add(module_instance.parameters())
        parameters_to_train = list(parameters_to_train)
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class(parameters_to_train, **optimizer_args)