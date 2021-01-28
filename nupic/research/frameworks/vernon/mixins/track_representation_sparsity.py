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

from pprint import pformat

from nupic.research.frameworks.pytorch.hooks import ModelHookManager, TrackSparsityHook
from nupic.research.frameworks.pytorch.model_utils import (
    filter_modules,
    get_module_attr,
)


class TrackRepresentationSparsity:
    """
    This mixin tracks and reports the average sparsities observed in the model's
    representations. Either input or output sparsities can be tracked for a specified,
    possibly overlapping, subset of modules. Tracked statistics are returned from
    `run_epoch` and are reset before each subsequent epoch. The default is to track none
    of the modules; only those specified can be tracked.

    .. _filter_modules: nupic.research.frameworks.pytorch.model_utils.filter_modules
    """

    def setup_experiment(self, config):
        """
        Register forward hooks to track the input and output sparsities.

        The subsets of modules to track are defined via `include_*` params. See
        `filter_modules`_ for further details.

        :param config:
            - track_input_sparsity_args:
                - include_modules: a list of module types to track
                - include_names: a list of module names to track e.g. "features.stem"
                - include_patterns: a list of regex patterns to compare to the names;
                                    for instance, all feature parameters in ResNet can
                                    be included through "features.*"
            - track_output_sparsity_args: same as track_input_sparsity_args

        .. _filter_modules: nupic.research.frameworks.pytorch.model_utils.filter_modules
        """
        super().setup_experiment(config)

        # Register hooks to track input sparsities.
        input_tracking_args = config.get("track_input_sparsity_args", {})
        named_modules = filter_modules(self.model, **input_tracking_args)
        self.input_hook_manager = ModelHookManager(named_modules, TrackSparsityHook)

        # Log the names of the modules with tracked inputs.
        tracked_names = pformat(list(named_modules.keys()))
        self.logger.info(f"Tracking input sparsity of modules: {tracked_names}")

        # Register hooks to track output sparsities.
        output_tracking_args = config.get("track_output_sparsity_args", {})
        named_modules = filter_modules(self.model, **output_tracking_args)
        self.output_hook_manager = ModelHookManager(named_modules, TrackSparsityHook)

        # Log the names of the modules with tracked outputs.
        tracked_names = pformat(list(named_modules.keys()))
        self.logger.info(f"Tracking output sparsity of modules: {tracked_names}")

        # Throw a warning when no modules are being tracked.
        if not input_tracking_args and not output_tracking_args:
            self.logger.warning("No modules specified to track input/output sparsity.")

    def run_epoch(self):
        """Run one epoch and log the observed sparsity statistics."""

        with self.input_hook_manager, self.output_hook_manager:
            ret = super().run_epoch()

            # Log sparsity statistics collected from input hooks.
            update_results_dict(ret, self.input_hook_manager, self.output_hook_manager)

        return ret

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        mixin = "TrackRepresentationSparsity: "
        eo["setup_experiment"].append(
            mixin + "Add forward hooks to track input and output sparsities.")
        eo["run_epoch"].append(
            mixin + "Calculate and log sparsity statistics of representations.")
        return eo


class TrackRepresentationSparsityMetaCL:
    """
    This mixin tracks and reports the average sparsities observed in the model's
    representations. This is similar to `TrackRepresentationSparsity` above, however,
    the functionality is reformatted for the MetaCL setting. Specifically, the hooks
    must be re-registered every epoch on the cloned model prior to the fast steps.

    Tracked statistics are returned from `run_epoch` and are reset before each
    subsequent epoch. The default is to track none of the modules; only those specified
    can be tracked.
    """

    def setup_experiment(self, config):
        """
        Save the names of which modules to track. The forward hooks are registered in
        `clone_model` at the start of every epoch of training.

        The subsets of modules to track are defined via `include_*` params. See
        `filter_modules`_ for further details.

        :param config:
            - track_input_sparsity_args:
                - include_modules: a list of module types to track
                - include_names: a list of module names to track e.g. "features.stem"
                - include_patterns: a list of regex patterns to compare to the names;
                                    for instance, all feature parameters in ResNet can
                                    be included through "features.*"
            - track_output_sparsity_args: same as track_input_sparsity_args

        .. _filter_modules: nupic.research.frameworks.pytorch.model_utils.filter_modules

        """
        super().setup_experiment(config)

        # The hooks managers will be initialized for each cloned model, prior to the
        # fast steps. For now, just the names of the modules to track will be saved.
        self.output_hook_manager = None
        self.input_hook_manager = None

        # Save the names of the modules to tracked inputs.
        input_tracking_args = config.get("track_input_sparsity_args", {})
        named_modules = filter_modules(self.model, **input_tracking_args)
        self.track_input_of_names = list(named_modules.keys())

        # Log the names of the modules with tracked inputs.
        tracked_names = pformat(self.track_input_of_names)
        self.logger.info(f"Tracking input sparsity of modules: {tracked_names}")

        # Save the names of the modules to tracked outputs.
        output_tracking_args = config.get("track_output_sparsity_args", {})
        named_modules = filter_modules(self.model, **output_tracking_args)
        self.track_output_of_names = list(named_modules.keys())

        # Log the names of the modules with tracked outputs.
        tracked_names = pformat(self.track_output_of_names)
        self.logger.info(f"Tracking output sparsity of modules: {tracked_names}")

        # Throw a warning when no modules are being tracked.
        if not input_tracking_args and not output_tracking_args:
            self.logger.warning("No modules specified to track input/output sparsity.")

    def run_epoch(self):
        """Run one epoch and log the observed sparsity statistics."""

        # The start of super's run_epoch, will call `clone_model` to initiate tracking.
        ret = super().run_epoch()

        # Log sparsity statistics collected from input hooks.
        update_results_dict(ret, self.input_hook_manager, self.output_hook_manager)

        return ret

    def clone_model(self):
        """
        Clones self.model and register forward hooks on new model.
        """
        model = super().clone_model()

        # Register hooks to track input sparsities.
        track_modules = get_modules_by_names(model, self.track_input_of_names)
        self.input_hook_manager = ModelHookManager(track_modules, TrackSparsityHook)

        # Register hooks to track output sparsities.
        track_modules = get_modules_by_names(model, self.track_output_of_names)
        self.output_hook_manager = ModelHookManager(track_modules, TrackSparsityHook)

        # Enable tracking the sparsity statistics.
        self.input_hook_manager.start_tracking()
        self.output_hook_manager.start_tracking()

        return model

    @classmethod
    def get_execution_order(cls):
        # TODO: Update eo.
        eo = super().get_execution_order()
        mixin = "TrackRepresentationSparsityMetaCL: "
        eo["setup_experiment"].append(
            mixin + "Save the names of which modules to track.")
        eo["clone_model"].append(
            mixin + "Register hooks to the cloned model.")
        eo["run_epoch"].append(
            mixin + "Calculate and log sparsity statistics of representations.")
        return eo


# -----------------
# Helper functions
# -----------------


def update_results_dict(results_dict, input_hook_manager, output_hook_manager):
    """Update a results dictionary with tracked sparsity statistics."""

    # Log sparsity statistics collected from input hooks.
    for name, module, x_sparsity, _ in input_hook_manager.get_statistics():

        cls_name = module.__class__.__name__
        if x_sparsity is not None:
            results_dict.update({
                f"input_sparsity {name} ({cls_name})": x_sparsity,
            })

    # Log sparsity statistics collected from output hooks.
    for name, module, _, y_sparsity in output_hook_manager.get_statistics():

        cls_name = module.__class__.__name__
        if y_sparsity is not None:
            results_dict.update({
                f"output_sparsity {name} ({cls_name})": y_sparsity,
            })


def get_modules_by_names(model, names):
    """Retrieve all modules of a model by name."""
    named_modules = {}
    for name in names:
        module = get_module_attr(model, name)
        named_modules[name] = module

    return named_modules


__all__ = [
    "TrackRepresentationSparsity",
    "TrackRepresentationSparsityMetaCL",
]
