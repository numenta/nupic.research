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

from contextlib import nullcontext
from pprint import pformat

from nupic.research.frameworks.pytorch.hooks import ModelHookManager, TrackSparsityHook


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
                - include_modules a list of module types to track
                - include_names a list of module names to track e.g. "features.stem"
                - include_patterns a list of regex patterns to compare to the names; for
                                   instance, all feature parameters in ResNet can be
                                   included through "features.*"
            - track_output_sparsity_args: same as track_input_sparsity_args

        .. _filter_modules: nupic.research.frameworks.pytorch.model_utils.filter_modules
        """
        super().setup_experiment(config)

        # By default, neither the inputs or output will be tracked.
        self.input_hook_manager = nullcontext()
        self.output_hook_manager = nullcontext()

        # Register hooks to track input sparsities.
        input_tracking_args = config.get("track_input_sparsity_args")
        if input_tracking_args is not None:

            # This manager will keep track of all input hooks.
            self.input_hook_manager = ModelHookManager(self.model, TrackSparsityHook,
                                                       **input_tracking_args)

            # Log the names of the tracked modules.
            tracked_modules = self.input_hook_manager.tracked_modules
            tracked_names = pformat(list(tracked_modules.keys()))
            self.logger.info(f"Tracking input sparsity of modules: {tracked_names}")

        # Register hooks to track output sparsities.
        output_tracking_args = config.get("track_output_sparsity_args")
        if output_tracking_args is not None:

            # This manager will keep track of all output hooks.
            self.output_hook_manager = ModelHookManager(self.model, TrackSparsityHook,
                                                        **output_tracking_args)

            # Log the names of the tracked modules.
            tracked_modules = self.output_hook_manager.tracked_modules
            tracked_names = pformat(list(tracked_modules.keys()))
            self.logger.info(f"Tracking output sparsity of modules: {tracked_names}")

        # Throw a warning when no modules are being tracked.
        if input_tracking_args is None and output_tracking_args is None:
            self.logger.warning("No modules specified to track input/output sparsity.")

    def run_epoch(self):
        """Run one epoch and log the observed sparsity statistics."""

        with self.input_hook_manager as xhooks, self.output_hook_manager as yhooks:
            ret = super().run_epoch()

            # Log sparsity statistics collected from input hooks.
            if xhooks is not None:
                for name, module, x_sparsity, _ in xhooks.get_statistics():

                    cls_name = module.__class__.__name__
                    if x_sparsity is not None:
                        ret.update({
                            f"input_sparsity {name} ({cls_name})": x_sparsity,
                        })

            # Log sparsity statistics collected from output hooks.
            if yhooks is not None:
                for name, module, _, y_sparsity in yhooks.get_statistics():

                    cls_name = module.__class__.__name__
                    if y_sparsity is not None:
                        ret.update({
                            f"output_sparsity {name} ({cls_name})": y_sparsity,
                        })

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
