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

from nupic.research.frameworks.dendrites import plot_contexts_by_class
from nupic.research.frameworks.pytorch.hooks import (
    ModelHookManager,
    TrackHiddenActivationsHook,
)
from nupic.research.frameworks.pytorch.model_utils import filter_modules

__all__ = [
    "PlotContextSignal"
]


class PlotContextSignal(metaclass=abc.ABCMeta):
    """
    Mixin for creating plots of the context vectors used to modulate dendritic
    networks.

    :param config: a dict containing the following

        - plot_context_args: a dict containing the following

            - include_modules: (optional) a list of module types to track
            - include_names: (optional) a list of module names to track e.g.
                             "features.stem"
            - include_patterns: (optional) a list of regex patterns to compare to the
                                names; for instance, all feature parameters in ResNet
                                can be included through "features.*"
            - plot_freq: (optional) how often to create the plot, measured in training
                         iterations; defaults to 1
            - max_samples_to_plot: (optional) how many of samples to use for plotting;
                                   only the newest will be used; defaults to 5000

    Example config:
    ```
    config=dict(
        plot_context_args=dict(
            include_names=["context_net"],
            plot_freq=1,
            max_samples_to_plot=2000
        ),
    )
    ```
    where `context_net` is assumed to refer to the module whose outputs are context
    vectors used to modulate a dendritic network.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Process config args
        context_args = config.get("plot_context_args", {})
        context_plot_freq, filter_args, max_samples = process_context_args(context_args)

        self.context_plot_freq = context_plot_freq
        self.context_max_samples = max_samples

        # Register hook for tracking context vectors
        named_modules = filter_modules(self.model, **filter_args)
        hook_args = dict(max_samples_to_track=self.context_max_samples)
        self.context_hook = ModelHookManager(named_modules,
                                             TrackHiddenActivationsHook,
                                             hook_args=hook_args)

        # Log the names of the modules being tracked
        tracked_names = pformat(list(named_modules.keys()))
        self.logger.info(f"Tracking context signals output from: {tracked_names}")

        # The targets will be collected in `self.error_loss` in a 1:1 fashion
        # to the tensors being collected by the hooks.
        self.context_targets = torch.tensor([]).long()

    def run_epoch(self):
        """
        This runs the epoch with the hooks in tracking mode. The resulting context
        vectors collected by the `TrackHiddenActivationsHook` object are plotted by
        calling a plotting function.
        """

        # Run the epoch with tracking enabled.
        with self.context_hook:
            results = super().run_epoch()

        # The epoch was iterated in `run_epoch` so epoch 0 is really epoch 1 here.
        iteration = self.current_epoch + 1

        # Create visualization, and update results dict.
        if iteration % self.context_plot_freq == 0:

            for name, _, contexts in self.context_hook.get_statistics():

                visual = plot_contexts_by_class(contexts, self.context_targets)
                results.update({f"contexts/{name}": visual})

        return results

    def error_loss(self, output, target, reduction="mean"):
        """
        This computes the loss and then saves the targets computed on this loss. This
        mixin assumes these targets correspond, in a 1:1 fashion, to the samples seen
        in the forward pass.
        """
        loss = super().error_loss(output, target, reduction=reduction)
        if self.context_hook.tracking:

            # Targets were initialized on the cpu which could differ from the
            # targets collected during the forward pass.
            self.context_targets = self.context_targets.to(target.device)

            # Concatenate and discard the older targets.
            self.context_targets = torch.cat([target, self.context_targets], dim=0)
            self.context_targets = self.context_targets[:self.context_max_samples]

        return loss


def process_context_args(context_args):

    context_args = deepcopy(context_args)

    # Collect information about which modules to apply hooks to
    include_names = context_args.pop("include_names", [])
    include_modules = context_args.pop("include_modules", [])
    include_patterns = context_args.pop("include_patterns", [])
    filter_args = dict(
        include_names=include_names,
        include_modules=include_modules,
        include_patterns=include_patterns,
    )

    # Others args
    plot_freq = context_args.get("plot_freq", 1)
    max_samples = context_args.get("max_samples_to_plot", 1000)

    assert isinstance(plot_freq, int)
    assert isinstance(max_samples, int)
    assert plot_freq > 0
    assert max_samples > 0

    return plot_freq, filter_args, max_samples
