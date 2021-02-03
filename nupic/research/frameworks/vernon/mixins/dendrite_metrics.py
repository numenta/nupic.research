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

import torch

from nupic.research.frameworks.dendrites import (
    ApplyDendritesHook,
    plot_mean_selected_activations,
)
from nupic.research.frameworks.pytorch.hooks import ModelHookManager
from nupic.research.frameworks.pytorch.model_utils import filter_modules

__all__ = [
    "TrackMeanSelectedActivations"
]


class TrackDendriteMetricsInterface(metaclass=abc.ABCMeta):
    """
    This in an interface for the mixins that track dendrite metrics.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        tracking_args = config.get(self.args_name, {})
        named_modules = filter_modules(self.model, **tracking_args)
        self.dendrite_hooks = ModelHookManager(named_modules,
                                               ApplyDendritesHook)

        self.targets = torch.tensor([]).long()

    @property
    @abc.abstractmethod
    def args_name(self):
        """
        The name of the args dict within config. These get passed to `filter_modules`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def plot_activations(self, dendrite_activations, winning_mask):
        """
        Plot activations according to the desired dendrite metric.
        """
        raise NotImplementedError

    def run_epoch(self):
        with self.dendrite_hooks:
            results = super().run_epoch()

        for name, _, activations, winning_mask in self.dendrite_hooks.get_statistics():
            # Plot stuff.
            visual = self.plot_activations(activations, winning_mask)
            results.update({f"mean_selected/{name}": visual})

        return results

    @property
    def tracking(self):
        return self.dendrite_hooks.tracking

    def error_loss(self, output, target, reduction="mean"):
        loss = super().error_loss(output, target, reduction=reduction)
        if self.tracking:
            self.targets = torch.cat([target, self.targets], dim=0)
            self.targets = self.targets[:100]
        return loss


class TrackMeanSelectedActivations(TrackDendriteMetricsInterface):
    """
    Track the mean selected activations of the `apply_dendrites` modules.

    :param config:
        - track_mean_selected_activations_args:
            - See `TrackDendriteMetricsInterface.setup_experiment()`
    """

    # The name of the args dict within config.
    args_name = "track_mean_selected_activations_args"

    def plot_activations(self, dendrite_activations, winning_mask):
        return plot_mean_selected_activations(dendrite_activations,
                                              winning_mask,
                                              self.targets)
