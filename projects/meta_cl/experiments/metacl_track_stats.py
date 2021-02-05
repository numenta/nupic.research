#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

from copy import deepcopy

import torch

from experiment_classes import OMLExperiment
from nupic.research.frameworks.dendrites import (
    DendriticAbsoluteMaxGate1d,
    plot_entropy_distribution,
)
from nupic.research.frameworks.vernon import mixins
from nupic.research.frameworks.wandb import ray_wandb

from .dendrites import DendritesExperiment, metacl_anml_dendrites
from .oml_replicate import metacl_oml_replicate


class TrackSparsityMetaCLExperiment(mixins.TrackRepresentationSparsity,
                                    OMLExperiment):
    pass


class TrackDendritesMetaCLExperiment(mixins.PlotDendriteMetrics,
                                     DendritesExperiment):
    pass


# AN example config using the TrackRepresentationSparsity mixin.
metacl_with_sparse_stats = deepcopy(metacl_oml_replicate)
metacl_with_sparse_stats.update(
    experiment_class=TrackSparsityMetaCLExperiment,

    # Track sparsity statistics.
    track_input_sparsity_args=dict(
        include_modules=[torch.nn.Linear]
    ),
    track_output_sparsity_args=dict(
        include_modules=[torch.nn.ReLU]
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="metacl_with_sparse_stats",
        project="test_metacl",
    ),
)


def plot_entropy_distribution_(_, winning_mask, targets):
    """
    This adjusts the function's signature to align with the mixin.
    """
    return plot_entropy_distribution(winning_mask, targets)


# AN example config using the PlotDendriteMetrics mixin.
metacl_with_dendrite_stats = deepcopy(metacl_anml_dendrites)
metacl_with_dendrite_stats.update(
    epochs=200,
    run_meta_test=False,
    experiment_class=TrackDendritesMetaCLExperiment,

    plot_dendrite_metrics_args=dict(
        include_modules=[DendriticAbsoluteMaxGate1d],
        entropy_distribution=dict(
            max_samples_to_plot=1000,
            plot_freq=50,
            # A wrapper will be placed around the function
            # to facilitate logging to wandb.
            plot_func=ray_wandb.prep_plot_for_wandb(
                plot_entropy_distribution_
            )
        )
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="metacl_with_dendrite_stats",
        project="test_metacl",
    ),
)

# Export configurations in this file
CONFIGS = dict(
    metacl_with_sparse_stats=metacl_with_sparse_stats,
    metacl_with_dendrite_stats=metacl_with_dendrite_stats,
)
