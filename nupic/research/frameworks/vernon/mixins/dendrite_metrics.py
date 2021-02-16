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

import torch

from nupic.research.frameworks.dendrites import ApplyDendritesBase, ApplyDendritesHook
from nupic.research.frameworks.pytorch.hooks import ModelHookManager
from nupic.research.frameworks.pytorch.model_utils import filter_modules

__all__ = [
    "PlotDendriteMetrics",
]


class PlotDendriteMetrics(metaclass=abc.ABCMeta):
    """
    This is a mixin for creating custom plots of metrics for
    apply-dendrite modules (those of type `ApplyDendritesBase`_). The user defines and
    gives a plotting function which is then called on the following arguments

        - dendrite_activations: the input activations passed to the apply-dendrites
                                module; these are meant to be the output of a
                                `DendriteSegments` module; they will be of shape
                                batch_size x num_units x num_segments
        - winning_mask: the mask of the winning segments, those chosen by the
                        apply-dendrites modules; this will be of shape
                        batch_size x num_units
        - targets: the targets that correspond to each sample in the batch

    Plots can be configured to use fewer samples (helpful for plotting a small batches
    of individual samples) and to plot every so many epochs (so that training isn't
    slowed down too much). Whenever a plot is made, the raw data used to create it is
    saved so it may be reproduced and edited off-line.

    .. warning:: When using this mixin with Ray, be careful to have 'plot_func' return
                 an object that can be logged. Often, Ray will attempt to create a
                 deepcopy prior to logging which can't be done on most plots. Try
                 wrapping the plot as done in `prep_plot_for_wandb`_.

    .. _ApplyDendritesBase: nupic/reasearch/frameworks/dendrites/modules
    .. _prep_plot_for_wandb: nupic/reasearch/frameworks/wandb/ray_wandb


    :param config: a dict containing the following

        - plot_dendrite_metrics_args: a dict containing the following

            - include_modules: (optional) a list of module types to track
            - include_names: (optional) a list of module names to track e.g.
                             "features.stem"
            - include_patterns: (optional) a list of regex patterns to compare to the
                                names; for instance, all feature parameters in ResNet
                                can be included through "features.*"

            <insert any plot name here>: This can be any string and maps to a dictionary
                                         of the plot arguments below. The
                                         resulting plot will be logged under
                                         "<plot_name>/<module_name>" in the results
                                         dictionary.

                - plot_func: the function called for plotting; must take three
                             arguments: 'dendrite_activations', 'winning_mask', and
                             'targets' (see above)
                - plot_freq: (optional) how often to create the plot, measured in
                             training iterations; defaults to 1
                - plot_args: (optional) either a dictionary or a callable that takes no
                             arguments and returns a dictionary; for instance this may
                             be used to return a random sample of integers specifying
                             units to plot; called only once at setup
                - max_samples_to_plot: (optional) how many of samples to use for
                                       plotting; only the newest will be used;
                                       defaults to 1000

    Example config:
    ```
    config=dict(
        plot_dendrite_metrics_args=dict(
            include_modules=[DendriticGate1d],
            mean_selected=dict(
                plot_func=plot_mean_selected_activations,
            )
        )
    )
    ```
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Unpack, validate, and process the default arguments.
        metric_args = config.get("plot_dendrite_metrics_args", {})
        self.metric_args, filter_args, max_samples = self.process_args(metric_args)

        # The maximum 'max_samples_to_track' will be tracked by the all the hooks.
        self.max_samples_to_track = max_samples
        hook_args = dict(max_samples_to_track=self.max_samples_to_track)

        # The 'filter_args' specify which modules to track.
        named_modules = filter_modules(self.model, **filter_args)
        self.dendrite_hooks = ModelHookManager(named_modules,
                                               ApplyDendritesHook,
                                               hook_args=hook_args)

        # The hook is specifically made for `ApplyDendritesBase` modules.
        for module in named_modules.values():
            assert isinstance(module, ApplyDendritesBase)

        # Log the names of the modules being tracked, and warn when there's none.
        names = list(named_modules.keys())
        self.logger.info(f"Dendrite Metric Setup: Tracking modules: {names}")
        if len(names) == 0:
            self.logger.warning("Dendrite Metric Setup: "
                                "No modules found for tracking.")

        # The targets will be collected in `self.error_loss` in a 1:1 fashion
        # to the tensors being collected by the hooks.
        self.targets = torch.tensor([]).long()

    def process_args(self, metric_args):

        metric_args = deepcopy(metric_args)

        # Remove and collect information about which modules to track.
        include_names = metric_args.pop("include_names", [])
        include_modules = metric_args.pop("include_modules", [])
        include_patterns = metric_args.pop("include_patterns", [])
        filter_args = dict(
            include_names=include_names,
            include_modules=include_modules,
            include_patterns=include_patterns,
        )

        # Gather and validate the metric arguments. The max of the 'max_samples_to_plot'
        # will be saved to dictate how many samples will be tracked by the hooks.
        all_max_num_samples = []
        new_metric_args = {}
        for metric_name, plotting_args in metric_args.items():

            plot_func = plotting_args.get("plot_func", None)
            plot_freq = plotting_args.get("plot_freq", 1)
            plot_args = plotting_args.get("plot_args", {})
            max_samples_to_plot = plotting_args.get("max_samples_to_plot", 1000)

            assert callable(plot_func)
            assert isinstance(plot_freq, int)
            assert isinstance(max_samples_to_plot, int)
            assert plot_freq > 0
            assert max_samples_to_plot > 0

            # The arguments may be given as a callable; useful for sampling random
            # values that dictate plotting to, say, only plot a subset of units
            if callable(plot_args):
                plot_args = plot_args()
            assert isinstance(plot_args, dict)

            new_metric_args[metric_name] = dict(
                plot_func=plot_func,
                plot_freq=plot_freq,
                plot_args=plot_args,
                max_samples_to_plot=max_samples_to_plot,
            )

            all_max_num_samples.append(max_samples_to_plot)

        max_samples_to_plot = max(all_max_num_samples)
        return new_metric_args, filter_args, max_samples_to_plot

    def run_epoch(self):
        """
        This runs the epoch with the hooks in tracking mode. The resulting 'activations'
        and 'winning_masks' collected by these hooks are plotted via each 'plot_func'
        along with their corresponding targets.
        """

        # Run the epoch with tracking enabled.
        with self.dendrite_hooks:
            results = super().run_epoch()

        # The epoch was iterated in `run_epoch` so epoch 0 is really epoch 1 here.
        iteration = self.current_epoch - 1

        # Gather and plot the statistics.
        for name, _, activations, winners in self.dendrite_hooks.get_statistics():

            # Keep track of whether a plot is made below. If so, save the raw data.
            plot_made = False

            # Each 'plot_func' will be applied to each module being tracked.
            for metric_name, plotting_args in self.metric_args.items():

                # All of the defaults were set in `process_args`.
                plot_func = plotting_args["plot_func"]
                plot_freq = plotting_args["plot_freq"]
                plot_args = plotting_args["plot_args"]
                max_samples_to_plot = plotting_args["max_samples_to_plot"]

                if iteration % plot_freq != 0:
                    continue

                # Only use up the the max number of samples for plotting.
                targets = self.targets[:max_samples_to_plot]
                activations = activations[:max_samples_to_plot]
                winners = winners[:max_samples_to_plot]

                # Call and log the results of the plot function.
                # Here, "{name}" is the name of the module.
                visual = plot_func(activations, winners, targets, **plot_args)
                results.update({f"{metric_name}/{name}": visual})
                plot_made = True

            # Log the raw data.
            if plot_made:
                targets = self.targets[:self.max_samples_to_track].cpu().numpy()
                activations = activations[:self.max_samples_to_track].cpu().numpy()
                winners = winners[:self.max_samples_to_track].cpu().numpy()
                results.update({f"targets/{name}": targets})
                results.update({f"dendrite_activations/{name}": activations})
                results.update({f"winning_mask/{name}": winners})

        return results

    def error_loss(self, output, target, reduction="mean"):
        """
        This computes the loss and then saves the targets computed on this loss. This
        mixin assumes these targets correspond, in a 1:1 fashion, to the images seen in
        the forward pass.
        """
        loss = super().error_loss(output, target, reduction=reduction)
        if self.dendrite_hooks.tracking:

            # Targets were initialized on the cpu which could differ from the
            # targets collected during the forward pass.
            self.targets = self.targets.to(target.device)

            # Concatenate and discard the older targets.
            self.targets = torch.cat([target, self.targets], dim=0)
            self.targets = self.targets[:self.max_samples_to_track]

        return loss
