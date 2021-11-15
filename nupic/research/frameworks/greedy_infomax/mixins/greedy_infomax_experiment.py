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
from nupic.research.frameworks.greedy_infomax.models.utility_layers import _PatchifyInputs
from nupic.research.frameworks.greedy_infomax.models.gim_model import GreedyInfoMaxModel
from nupic.research.frameworks.greedy_infomax.utils.data_utils import patchify_inputs
from nupic.research.frameworks.vernon import mixins, SelfSupervisedExperiment
from torch.utils.data import DataLoader
from nupic.research.frameworks.greedy_infomax.utils.train_utils import (
    aggregate_eval_results_gim,
    evaluate_gim_model,
    train_gim_model,
)
import copy


class GreedyInfoMaxExperiment(
    mixins.LogEveryLoss,
    mixins.LogEveryLearningRate,
    mixins.LogBackpropStructure,
    mixins.RezeroWeights,
    SelfSupervisedExperiment,
):
    """
    GreedyInfoMaxExperiment is a subclass of SelfSupervisedExperiment that wraps the
    model passed into the config in a GreedyInfoMaxModel and adds the necessary
    modules at specific points in the model's forward pass. It does all of the
    following:

    1. Adds a _PatchifyInputs module at the beginning of the model
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
            -patchify_inputs_args: a dict containing the following
                - patch_size: the size of the patches to use for the model (defaults
                to 5)
                - overlap: the amount of overlap between patches (defaults to 2)

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
            patchify_inputs_args=dict(
                patch_size=16,
                overlap=2
            )
        ),
    )
    ```
    """

    def setup_experiment(self, config):
        super(GreedyInfoMaxExperiment, self).setup_experiment(config)
        self.evaluate_model = evaluate_gim_model
        self.train_model = self.train_model_supervised = train_gim_model
        self.multiple_module_loss_history = []


    def create_model(self, config, device):
        sample_data = self.get_sample_data(config)
        model = super().create_model(config, device)
        # Process config args
        greedy_infomax_args = config.get("greedy_infomax_args", {})
        gim_hooks_args = greedy_infomax_args.get("greedy_infomax_blocks", {})
        info_estimate_args = greedy_infomax_args.get("info_estimate_args", {})
        patchify_inputs_args = greedy_infomax_args.get("patchify_inputs_args", {})

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
        named_modules = filter_modules(model, **filter_args)

        # Get the size of the output of each module (for BilinearInfo)
        modules_and_channel_sizes = get_channel_sizes(model,
                                                      named_modules,
                                                      sample_data)
        # Update the config with the channel sizes
        config["classifier_config"]["model_args"].update(
            in_channels=list(modules_and_channel_sizes.values())
        )

        n_patches_x, n_patches_y = get_patch_dimensions(sample_data, **patchify_inputs_args)

        greedy_infomax_model = GreedyInfoMaxModel(model,
                                        modules_and_channel_sizes,
                                        **info_estimate_args,
                                        n_patches_x=n_patches_x,
                                        n_patches_y=n_patches_y)

        return greedy_infomax_model

    def get_sample_data(self, config):
        """
        Get a sample data dictionary for the model.
        """
        sample_dataset = self.load_dataset(config, dataset_type="supervised")
        sample_dataloader = DataLoader(
            dataset=sample_dataset,
            batch_size=2,
        )
        sample_data, _ = next(iter(sample_dataloader))
        return sample_data

    def post_batch(self, error_loss, complexity_loss, batch_idx, **kwargs):
        super().post_batch(
            error_loss=error_loss,
            complexity_loss=complexity_loss,
            batch_idx=batch_idx,
            **kwargs,
        )
        if self.should_log_batch(batch_idx) and "module_losses" in kwargs.keys():
            self.multiple_module_loss_history.append(kwargs["module_losses"].clone())

    def run_epoch(self):
        result = super().run_epoch()
        if len(self.multiple_module_loss_history) > 0:
            log = torch.stack(self.multiple_module_loss_history)
            module_loss_history = log.cpu().numpy()
            for i in range(log.shape[1]):
                result[f"module_{i}_loss_history"] = module_loss_history[:, i].tolist()
            self.multiple_module_loss_history = []
            result["num_bilinear_info_modules"] = int(log.shape[1])
        return result

    @classmethod
    def get_readable_result(cls, result):
        return result

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        result_by_timestep = super().expand_result_to_time_series(result, config)
        recorded_timesteps = cls.get_recorded_timesteps(result, config)
        for i in range(result["num_bilinear_info_modules"]):
            for t, loss in zip(recorded_timesteps, result[f"module_{i}_loss_history"]):
                result_by_timestep[t].update({f"module_{i}_train_loss": loss})
        return result_by_timestep

    @classmethod
    def _aggregate_validation_results(cls, results):
        result = copy.copy(results[0])
        result.update(aggregate_eval_results_gim(results))
        return result

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("GreedyInfoMaxExperiment: initialize")
        eo["post_batch"].append("GreedyInfoMaxExperiment: record losses")
        eo["run_epoch"].append("GreedyInfoMaxExperiment: to result dict")
        eo["expand_result_to_time_series"].append("GreedyInfoMaxExperiment: module_losses")
        return eo


def get_channel_sizes(model, named_modules, sample_data):
    """
    Get the size of the output of each module (for BilinearInfo). Returns a
    dictionary of the form {module_name: output_size}
    """
    modules_and_channel_sizes = {}
    for module_name, module in named_modules.items():
        # attach a hook to each module in the model that needs a size calculated
        def module_size_hook(module, input, output):
            modules_and_channel_sizes[module] = output.size()[1] # b,c,h,w
        module.register_forward_hook(module_size_hook)
    # do a single forward pass through the model
    model(sample_data)
    # detach the hook
    for module in named_modules.values():
        module._forward_hooks.clear()
    return modules_and_channel_sizes

def get_patch_dimensions(sample_data, patch_size=16, overlap=2):
    x, n_patches_x, n_patches_y = patchify_inputs(sample_data, patch_size, overlap)
    return n_patches_x, n_patches_y