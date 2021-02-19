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

import abc
from copy import deepcopy
from pprint import pformat

import torch

from nupic.research.frameworks.dendrites import (
    plot_repr_overlap_distributions,
    plot_repr_overlap_matrix,
)
from nupic.research.frameworks.pytorch.hooks import (
    ModelHookManager,
    TrackHiddenActivationsHook,
)
from nupic.research.frameworks.pytorch.model_utils import filter_modules

__all__ = [
    "PlotRepresentationOverlap",
]


class PlotRepresentationOverlap(metaclass=abc.ABCMeta):
    """
    Mixin for plotting a module's inter- and intra-class representation overlap.

    :param config: a dict containing the following

        - plot_representation_overlap_args: a dict containing the following

            - include_modules: (optional) a list of module types to track
            - include_names: (optional) a list of module names to track e.g.
                             "features.stem"
            - include_patterns: (optional) a list of regex patterns to compare to the
                                names; for instance, all feature parameters in ResNet
                                can be included through "features.*"
            - plot_freq: (optional) how often to create the plot, measured in training
                         iterations; defaults to 1
            - plot_args: (optional) either a dictionary or a callable that takes no
                             arguments and returns a dictionary; for instance this may
                             be used to return a random sample of integers specifying
                             units to plot; called only once at setup
            - max_samples_to_plot: (optional) how many of samples to use for plotting;
                                   only the newest will be used; defaults to 5000

    Example config:
    ```
    config=dict(
        plot_representation_overlap_args=dict(
            include_modules=[torch.nn.ReLU, KWinners],
            plot_freq=1,
            plot_args=dict(annotate=False),
            max_samples_to_plot=2000
        )
    )
    ```
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Process config args
        ro_args = config.get("plot_representation_overlap_args", {})
        ro_plot_freq, filter_args, ro_max_samples = self.process_ro_args(ro_args)

        self.ro_plot_freq = ro_plot_freq
        self.ro_max_samples = ro_max_samples

        # Register hook for tracking hidden activations - useful for representation
        # overlap
        named_modules = filter_modules(self.model, **filter_args)
        hook_args = dict(max_samples_to_track=self.ro_max_samples)
        self.ro_hook = ModelHookManager(named_modules,
                                        TrackHiddenActivationsHook,
                                        hook_args=hook_args)

        # Log the names of the modules being tracked
        tracked_names = pformat(list(named_modules.keys()))
        self.logger.info(f"Tracking representation overlap of modules: {tracked_names}")

        # The targets will be collected in `self.error_loss` in a 1:1 fashion
        # to the tensors being collected by the hooks.
        self.ro_targets = torch.tensor([]).long()

    def process_ro_args(self, ro_args):

        ro_args = deepcopy(ro_args)

        # Collect information about which modules to apply hooks to
        include_names = ro_args.pop("include_names", [])
        include_modules = ro_args.pop("include_modules", [])
        include_patterns = ro_args.pop("include_patterns", [])
        filter_args = dict(
            include_names=include_names,
            include_modules=include_modules,
            include_patterns=include_patterns,
        )

        # Others args
        plot_freq = ro_args.get("plot_freq", 1)
        plot_args = ro_args.get("plot_args", {})
        max_samples = ro_args.get("max_samples_to_plot", 1000)

        assert isinstance(plot_freq, int)
        assert isinstance(plot_args, dict)
        assert isinstance(max_samples, int)
        assert plot_freq > 0
        assert max_samples > 0

        return plot_freq, filter_args, max_samples

    def run_epoch(self):

        # Run the epoch with tracking enabled.
        with self.ro_hook:
            results = super().run_epoch()

        # The epoch was iterated in `run_epoch` so epoch 0 is really epoch 1 here.
        iteration = self.current_epoch + 1

        # Create visualization, and update results dict.
        if iteration % self.ro_plot_freq == 0:

            for name, _, activations in self.ro_hook.get_statistics():

                # Representation overlap matrix
                visual = plot_repr_overlap_matrix(activations, self.ro_targets)
                results.update({f"repr_overlap_matrix/{name}": visual})

                # Representation overlap distributions:
                #  * inter-class pairs
                #  * intra-class pairs
                inter_ol, intra_ol = plot_repr_overlap_distributions(activations,
                                                                     self.ro_targets)
                results.update({f"repr_overlap_interclass/{name}": inter_ol})
                results.update({f"repr_overlap_intraclass/{name}": intra_ol})

        return results

    def error_loss(self, output, target, reduction="mean"):
        """
        This computes the loss and then saves the targets computed on this loss. This
        mixin assumes these targets correspond, in a 1:1 fashion, to the samples seen
        in the forward pass.
        """
        loss = super().error_loss(output, target, reduction=reduction)
        if self.ro_hook.tracking:

            # Targets were initialized on the cpu which could differ from the
            # targets collected during the forward pass.
            self.ro_targets = self.ro_targets.to(target.device)

            # Concatenate and discard the older targets.
            self.ro_targets = torch.cat([target, self.ro_targets], dim=0)
            self.ro_targets = self.ro_targets[:self.ro_max_samples]

        return loss
