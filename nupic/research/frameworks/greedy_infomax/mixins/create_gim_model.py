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

from copy import deepcopy
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nupic.research.frameworks.pytorch.model_utils import filter_modules
from nupic.research.frameworks.greedy_infomax.models.gim_block import \
    GreedyInfoMaxBlock, InfoEstimateAggregator, EncodingAggregator
from nupic.research.frameworks.greedy_infomax.models.utility_layers import PatchifyInputs
from nupic.research.frameworks.greedy_infomax.models.gim_model import GIMModel


class GreedyInfoMaxModel:
    """
    Mixin for running a Greedy InfoMax experiment. This mixin does multiple things:
    1. Adds a PatchifyInputs module at the beginning of the model
    2. Adds a GreedyInfoMaxBlock module to any specified modules
    3. Adds an EncodingAggregator module to the model
    4. Adds an InfoEstimateAggregator module to the model

    :param config: a dict containing the following
        - create_gim_model_args: a dict containing the following
            - gim_hooks_args: a list of module names to be added to the model
                - include_modules: (optional) a list of module types to track
                - include_names: (optional) a list of module names to track e.g.
                                 "features.stem"
                - include_patterns: (optional) a list of regex patterns to compare to the
                                    names; for instance, all feature parameters in ResNet
                                    can be included through "features.*"
            - info_estimate_args: a dict containing the following
                - k_predictions: the number of predictions to use or each info
                estimate block (defautls to 5)
                - negative_samples: the number of negative samples to use for each
                info estimate block (defaults to 16)

    Example config:
    ```
    config=dict(
        create_gim_model_args=dict(
            gim_hooks_args=dict(
                include_modules=[torch.nn.Conv2d, KWinners],
                include_names=["features.stem", "features.stem.kwinners"],
                include_patterns=["features.*"]
            ),
            info_estimate_args=dict(
                k_predictions=5,
                negative_samples=16
            ),
        ),
    )
    ```
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)
        # Process config args
        create_gim_model_args = config.get("create_gim_model_args", {})
        gim_hooks_args = create_gim_model_args.get("gim_hooks_args", {})
        info_estimate_args = create_gim_model_args.get("info_estimate_args", {})


        # Collect information about which modules to apply hooks to
        include_names = gim_hooks_args.pop("include_names", [])
        include_modules = gim_hooks_args.pop("include_modules", [])
        include_patterns = gim_hooks_args.pop("include_patterns", [])
        filter_args = dict(
            include_names=include_names,
            include_modules=include_modules,
            include_patterns=include_patterns,
        )

        # Get named modules for GreedyInfoMaxBlock and BilinearInfo parameters
        named_modules = filter_modules(self.model, **filter_args)

        model = self.model
        self.model = GIMModel(model, named_modules, **info_estimate_args)


