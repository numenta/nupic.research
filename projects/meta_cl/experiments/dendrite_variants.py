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

import torch

from experiment_classes import BoostedDendritesExperiment
from networks import BoostedANMLDendriticNetwork, ReplicateANMLDendriticNetwork
from nupic.research.frameworks.dendrites import plot_winning_segment_distributions
from nupic.research.frameworks.vernon import mixins
from nupic.research.frameworks.wandb import ray_wandb

from .dendrites import metacl_anml_dendrites_adjust_lr


class TrackBoostedDendritesExperiment(mixins.TrackRepresentationSparsity,
                                      mixins.PlotDendriteMetrics,
                                      BoostedDendritesExperiment):
    pass


# This is highly similar to the ANML model, but with a multi-segmented dendritic layer.
# Note, in contrast, `anml_replicated_with_dendrites` is identical to ANML in that it
# only has one segment.
#
# This model uses 5,962,272 weights on out of 26,885,763 whereas ANML uses 5,963,139
# weights and is fully dense.
#

def get_plot_args():
    """Arguments for plot_winning_segment_distributions"""
    return dict(
        num_units_to_plot=10,
        seed=torch.initial_seed()
    )


# Wrap function to make output plot wandb compatible.
@ray_wandb.prep_plot_for_wandb
def plot_winning_segment_distributions_wandb(
    dendrite_activations_,
    winning_mask,
    targets_,
    **kwargs
):
    # Adjust signature to work with `PlotDendriteMetrics` mixin.
    return plot_winning_segment_distributions(winning_mask, **kwargs)


# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.85 ± 0.04      | 0.94 ± 0.03       | 0.001 |
# |            50 | 0.84 ± 0.02      | 0.98 ± 0.01       | 0.001 |
# |           100 | 0.81 ± 0.02      | 0.98 ± 0.00       | 0.001 |
# |           200 | 0.76 ± 0.02      | 0.98 ± 0.00       | 0.001 |
# |           600 | 0.62 ± 0.01      | 0.96 ± 0.00       | 0.001 |
# |--------------------------------------------------------------|
#
anml_dendrites_multi_segments = deepcopy(metacl_anml_dendrites_adjust_lr)
anml_dendrites_multi_segments.update(
    experiment_class=TrackBoostedDendritesExperiment,
    model_class=ReplicateANMLDendriticNetwork,
    model_args=dict(
        num_classes=963,
        num_segments=10,
        dendrite_sparsity=0.9008,
        dendrite_bias=True
    ),

    # Track sparsity statistics.
    track_input_sparsity_args=dict(
        include_modules=[torch.nn.Linear]
    ),

    # Plotting args.
    plot_dendrite_metrics_args=dict(
        winning_segments=dict(
            plot_func=plot_winning_segment_distributions_wandb,
            plot_freq=50,
            plot_args=get_plot_args,
            max_samples_to_track=2000,
        )
    ),

    wandb_args=dict(name="anml_dendrites_multi_segments", project="metacl"),
)


# |--------------:|:-----------------|:------------------|-------:|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.84 ± 0.03      | 0.92 ± 0.03       | 0.001 |
# |            50 | 0.85 ± 0.03      | 0.99 ± 0.01       | 0.001 |
# |           100 | 0.81 ± 0.03      | 0.98 ± 0.01       | 0.001 |
# |           200 | 0.76 ± 0.02      | 0.98 ± 0.01       | 0.001 |
# |           600 | 0.62 ± 0.01      | 0.96 ± 0.00       | 0.001 |
# |---------------------------------------------------------------|
#
anml_dendrites_multi_segments_boosted = deepcopy(anml_dendrites_multi_segments)
anml_dendrites_multi_segments_boosted.update(
    experiment_class=TrackBoostedDendritesExperiment,
    model_class=BoostedANMLDendriticNetwork,

    model_args=dict(
        num_classes=963,
        num_segments=10,
        dendrite_sparsity=0.9008,
        dendrite_bias=True,
        boost_strength_factor=0.995,
    ),

    # Track sparsity statistics.
    track_input_sparsity_args=dict(
        include_modules=[torch.nn.Linear]
    ),

    # Plotting args.
    plot_dendrite_metrics_args=dict(
        winning_segments=dict(
            plot_func=plot_winning_segment_distributions_wandb,
            plot_freq=50,
            plot_args=get_plot_args,
            max_samples_to_track=2000,
        )
    ),

    wandb_args=dict(name="anml_dendrites_multi_segments_boosted", project="metacl"),
)


# Run the above but with a high boost strength. This has the same decay rate over time
# as the of `default_sparse_cnn` in the GSC projects folder.
# |---------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |     LR |
# |--------------:|:-----------------|:------------------|-------:|
# |            10 | 0.88 ± 0.03      | 0.94 ± 0.03       | 0.001  |
# |            50 | 0.87 ± 0.02      | 0.98 ± 0.01       | 0.001  |
# |           100 | 0.83 ± 0.03      | 0.97 ± 0.02       | 0.001  |
# |           200 | 0.76 ± 0.01      | 0.98 ± 0.00       | 0.0006 |
# |           600 | 0.63 ± 0.00      | 0.94 ± 0.00       | 0.001  |
# |---------------------------------------------------------------|
#
anml_dendrites_segments_boosted_09998 = deepcopy(anml_dendrites_multi_segments_boosted)
anml_dendrites_segments_boosted_09998.update(
    model_args=dict(
        num_classes=963,
        num_segments=10,
        dendrite_sparsity=0.9008,
        dendrite_bias=True,
        boost_strength_factor=0.9998419717142984,
    ),

    wandb_args=dict(
        name="anml_dendrites_segments_boosted_09998",
        project="metacl",
    ),
)


CONFIGS = dict(
    anml_dendrites_multi_segments=anml_dendrites_multi_segments,
    anml_dendrites_multi_segments_boosted=anml_dendrites_multi_segments_boosted,
    anml_dendrites_segments_boosted_09998=anml_dendrites_segments_boosted_09998,
)
