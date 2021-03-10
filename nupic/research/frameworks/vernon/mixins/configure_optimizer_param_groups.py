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

import warnings
from pprint import pformat

import torch

from nupic.research.frameworks.pytorch.model_utils import filter_params


class ConfigureOptimizerParamGroups:
    """
    This mixin enables configurable optimizer arguments for user specified param groups.
    """

    def setup_experiment(self, config):
        """
        Log param groups after creating optimizer.
        """
        super().setup_experiment(config)

        # The last group is the default group, so it won't be included.
        param_groups = self.optimizer.param_groups[:-1]

        # Print the param names and optimizer arguments belonging to each group.
        for g, param_group in enumerate(param_groups):

            # The `_name` attribute was set in `create_optimizer`.
            params = param_group["params"]
            names = [param._name for param in params if hasattr(param, "_name")]
            args = {key: val for key, val in param_group.items() if key != "params"}

            self.logger.info(f"\nParam group {g}:"
                             f"\n   Params in group: \n{pformat(names, indent=6)}"
                             f"\n   Group optim args: \n{pformat(args, indent=6)}")

    @classmethod
    def create_optimizer(cls, config, model):
        """
        Create optimizer with param groups as specified in experiment config.

        :param config:
            - optimizer_class: Callable or class to instantiate optimizer. Must return
                               object inherited from "torch.optim.Optimizer"
            - optimizer_args: Default arguments to use for the non specific param groups
            - optim_args_groups: a list of dictionaries, one for each param group;
                                 includes the following arguments:
                - group_args: dict of arguments for that param group (e.g. {"lr": 0.1})
                - include_modules: (optional) a list of module types; params belonging
                                   to these module types will be included in the group;
                                   for example, [torch.nn.Linear]
                - include_names: (optional) a list of param names to include in the
                                 group; for example, ["stem.weight", "stem.bias"]
                - include_patterns: (optional) a list of regex patterns to check
                                    against the names; those that match will be
                                    included in the group; for example, ["batch_norm.*"]

        Example config:
        ```
        from torch.nn.modules.batchnorm import _BatchNorm

        config=dict(
            optim_args_groups=[

                # Group 0: Stem - reduced lr
                dict(
                    group_args=dict(lr=0.001),
                    include_names=["features.stem.weight"],
                ),

                # Group 1: Batchnorm and Bias - no weight decay
                dict(
                    group_args=dict(weight_decay=0),
                    include_patterns=[".*bias"],
                    include_modules=[_BatchNorm],
                ),
            ]
        )
        ```
        """

        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        default_optim_args = config.get("optimizer_args", {})
        grouped_optim_args = []

        # Keep track of the params used across all groups to later recover the remaining
        used_params = []

        for group in config.get("optim_args_groups", []):

            # Get the param group arguments (e.g. lr, weight_decay)
            assert "group_args" in group
            group_args = group["group_args"]

            # Filter out group specific params.
            include_modules = group.get("include_modules", [])
            include_names = group.get("include_names", [])
            include_patterns = group.get("include_patterns", [])
            named_params = filter_params(model,
                                         include_modules=include_modules,
                                         include_names=include_names,
                                         include_patterns=include_patterns)

            # Set the `_name` of parameters. This is a new attribute, and will be
            # accessed in `setup_experiment` for logging purposes.
            for name, param in named_params.items():
                param._name = name

            params = list(named_params.values())
            used_params.extend([p.data_ptr() for p in params])

            # Raise a warning if the `include*` arguments filter down to an empty set.
            if len(params) == 0:
                warnings.warn("Unable to find any params for given group:\n"
                              f"{group}")

            # Add param group to list.
            grouped_optim_args.append(
                dict(
                    params=params,
                    **group_args
                )
            )

        # Collect remaining unused params.
        remaining = []
        for param in model.parameters():
            if param.data_ptr() not in used_params:
                remaining.append(param)
        grouped_optim_args.append(dict(params=remaining))

        # Instantiate optimizer.
        return optimizer_class(grouped_optim_args, **default_optim_args)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Log param groups.")
        eo["create_optimizer"] = ["Configure optimizer param groups."]
        return eo
