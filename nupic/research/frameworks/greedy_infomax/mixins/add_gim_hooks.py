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

from nupic.research.frameworks.pytorch.hooks import ModelHookManager, TrackGradientsHook
from nupic.research.frameworks.pytorch.model_utils import filter_modules


class AddGIMHooks(object):
    """
    Mixin for adding hooks to arbitrary modules in a model for running a Greedy
    InfoMax experiment.


    :param config: a dict containing the following
        - gim_hooks_args: a dict containing the following
            - include_modules: (optional) a list of module types to track
            - include_names: (optional) a list of module names to track e.g.
                             "features.stem"
            - include_patterns: (optional) a list of regex patterns to compare to the
                                names; for instance, all feature parameters in ResNet
                                can be included through "features.*"

    Example config:
    ```
    config=dict(
        gim_hooks_args=dict(
            include_modules=[torch.nn.Linear, KWinners],
            plot_freq=1,
            max_samples_to_track=150,
            metrics=["dot", "pearson"],
            gradient_values="mask",

            metric1 = "mask/dot"
            metric2 = "sign/pearson"
        )
    )
    ```
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Process config args
        gradient_metrics_args = config.get("gradient_metrics_args", {})
        self.gradient_metrics_plot_freq, self.gradient_metrics_filter_args, \
            self.gradient_metrics_max_samples, self.gradient_metrics = \
            process_gradient_metrics_args(
                gradient_metrics_args)

        # Register hook for tracking hidden activations
        named_modules = filter_modules(self.model, **self.gradient_metrics_filter_args)
        hook_args = dict(max_samples_to_track=self.gradient_metrics_max_samples)
        self.gradient_metric_hooks = ModelHookManager(
            named_modules, TrackGradientsHook, hook_type="backward", hook_args=hook_args
        )

        # Log the names of the modules being tracked
        tracked_names = pformat(list(named_modules.keys()))
        self.logger.info(f"Tracking gradients for modules: {tracked_names}")

        # The targets will be collected in `self.error_loss` in a 1:1 fashion
        # to the tensors being collected by the hooks.
        self.gradient_metric_targets = torch.tensor([]).long()

    def run_epoch(self):
        """
        This runs the epoch with the hooks in tracking mode. The resulting gradients
        collected by the `TrackGradientsHook` object are plotted by calling a
        plotting function.
        """

        # Run the epoch with tracking enabled.
        with self.gradient_metric_hooks:
            results = super().run_epoch()

        # The epoch was iterated in `run_epoch` so epoch 0 is really epoch 1 here.
        iteration = self.current_epoch + 1

        # Calculate metrics, create visualization, and update results dict.
        if iteration % self.gradient_metrics_plot_freq == 0:
            gradient_stats = self.gradient_metric_hooks.get_statistics()
            gradient_metrics_stats = self.calculate_gradient_metrics_stats(
                gradient_stats
            )
            gradient_metric_heatmaps = self.plot_gradient_metric_heatmaps(
                gradient_metrics_stats
            )
            for (name, _, gradient_metric, gradient_value, _, figure) in \
                    gradient_metric_heatmaps:
                results.update({f"{gradient_metric}/{gradient_value}/{name}": figure})
        return results

    def calculate_gradient_metrics_stats(self, gradients_stats):
        """
        This function calculates statistics given the gradients_stats which are being
        tracked by the TrackGradients backwards hook.

        This function accesses self.metrics, which is alsa a list of tuples. Each
        tuple is the combination of a metric function ("cosine", "dot", "pearson)
        and a gradient transformation ("real", "sign", "mask"). By default,
        the statistics that are calculated are ("cosine", "mask") and ("cosine",
        "real"). ("cosine", "mask") corresponds to the overlap between two different
        gradients, and ("cosine", "real") corresponds to the standard cosine
        similarity between two gradients.

        Args:
            gradients_stats: A list of tuples, each of which contains a named module
            and its gradients

        Returns:
            A list of tuples, each of which contains a named module, the statistics
            calculated from its gradients, and the metric function/transformation
            used on the gradients
        """
        all_stats = []
        for (name, module, gradients) in gradients_stats:
            for gradient_metric, gradient_value in self.gradient_metrics:
                # apply gradient value transformation if necessary
                if gradient_value == "sign":
                    gradients = torch.sign(gradients)
                elif gradient_value == "mask":
                    gradients = torch.abs(torch.sign(gradients))
                # calculate metric function on transformed gradients
                if gradient_metric == "cosine":
                    stats = [
                        torch.cosine_similarity(x, y, dim=0)
                        if not torch.equal(x, y) else 0
                        for x in gradients
                        for y in gradients
                    ]
                elif gradient_metric == "dot":
                    stats = [x.dot(y) if not torch.equal(x, y) else 0
                             for x in gradients
                             for y in gradients]
                elif gradient_metric == "pearson":
                    stats = [
                        torch.cosine_similarity(x - x.mean(), y - y.mean(), dim=0)
                        if not torch.equal(x, y) else 0
                        for x in gradients
                        for y in gradients
                    ]
                stats = torch.tensor(stats)
                gradient_dim = len(gradients)
                stats = stats.view(gradient_dim, gradient_dim)
                all_stats.append((name, module, gradient_metric, gradient_value, stats))
        return all_stats

    def plot_gradient_metric_heatmaps(self, gradient_metrics_stats):
        order_by_class = torch.argsort(self.gradient_metric_targets)
        sorted_gradient_metric_targets = self.gradient_metric_targets[order_by_class]
        class_change_indices = \
            (sorted_gradient_metric_targets - sorted_gradient_metric_targets.roll(
                1)).nonzero(as_tuple=True)[0].cpu()
        class_labels = [int(_x) for _x in
                        sorted_gradient_metric_targets.unique()]
        class_change_indices_right = class_change_indices.roll(-1).cpu()
        class_change_indices_right[-1] = len(sorted_gradient_metric_targets)
        tick_locations = (class_change_indices + class_change_indices_right) / 2.0 - 0.5
        tick_locations = tick_locations.cpu()

        stats_and_figures = []
        for (name, module, gradient_metric, gradient_value, stats) in \
                gradient_metrics_stats:
            stats = stats[order_by_class, :][:, order_by_class]
            ax = plt.gca()
            max_val = np.abs(stats).max()
            img = ax.imshow(stats, cmap="bwr", vmin=-max_val, vmax=max_val)
            ax.set_xlabel("class")
            ax.set_ylabel("class")
            for idx in class_change_indices:
                ax.axvline(idx - 0.5, color="black")
                ax.axhline(idx - 0.5, color="black")
            ax.set_xticks(tick_locations)
            ax.set_yticks(tick_locations)
            ax.set_xticklabels(class_labels)
            ax.set_yticklabels(class_labels)
            ax.set_title(f"{gradient_metric}:{gradient_value}:{name}")
            plt.colorbar(img, ax=ax)
            plt.tight_layout()
            figure = plt.gcf()
            stats_and_figures.append((name, module, gradient_metric,
                                      gradient_value, stats, figure))
        return stats_and_figures

    def error_loss(self, output, target, reduction="mean"):
        """
        This computes the loss and then saves the targets computed on this loss. This
        mixin assumes these targets correspond, in a 1:1 fashion, to the samples seen
        in the forward pass.
        """
        loss = super().error_loss(output, target, reduction=reduction)
        if self.gradient_metric_hooks.tracking:

            # Targets were initialized on the cpu which could differ from the
            # targets collected during the forward pass.
            self.gradient_metric_targets = self.gradient_metric_targets.to(
                target.device
            )

            # Concatenate and discard the older targets.
            self.gradient_metric_targets = torch.cat(
                [target, self.gradient_metric_targets], dim=0
            )
            self.gradient_metric_targets = self.gradient_metric_targets[
                : self.gradient_metrics_max_samples
            ]

        return loss


def process_gradient_metrics_args(gradient_metric_args):

    gradient_metrics_args = deepcopy(gradient_metric_args)

    # Collect information about which modules to apply hooks to
    include_names = gradient_metrics_args.pop("include_names", [])
    include_modules = gradient_metrics_args.pop("include_modules", [])
    include_patterns = gradient_metrics_args.pop("include_patterns", [])
    filter_args = dict(
        include_names=include_names,
        include_modules=include_modules,
        include_patterns=include_patterns,
    )

    # Others args
    plot_freq = gradient_metrics_args.get("plot_freq", 1)
    max_samples = gradient_metrics_args.get("max_samples_to_track", 100)
    metrics = gradient_metrics_args.get("metrics", [("cosine", "mask"),
                                                    ("cosine", "real")])

    available_metrics_options = ["cosine", "pearson", "dot"]
    available_gradient_values_options = ["real", "sign", "mask"]

    assert isinstance(plot_freq, int)
    assert isinstance(max_samples, int)
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    assert all([metric in available_metrics_options
                and gradient_value in available_gradient_values_options
                for metric, gradient_value in metrics])
    assert plot_freq > 0
    assert max_samples > 0

    return plot_freq, filter_args, max_samples, metrics
