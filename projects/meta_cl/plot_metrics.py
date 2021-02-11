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

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def get_random_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def plot_winning_segment_distribution(winning_indices, num_segments, unit=0, ax=None):
    binrange = (0, num_segments)
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(
        winning_indices,
        kde=True,
        stat="probability",
        binwidth=1,
        binrange=binrange,
        ax=ax,
    )
    ax.set_xlabel("Segment")
    ax.set_title(f"Probability of Activation of Unit {unit}")
    return ax


def plot_winning_segment_distributions(
    dendrite_activations_,
    winning_mask,
    targets_,
    num_units_to_plot=1,
    seed=0
):
    """
    Plot the distribution of winning segments for the list of units (defaults to just
    the first):

    :param dendrite_activations_: unused
    :param winning_mask: the winning mask of segments;
                         shape num_samples x num_units x num_segments
    :param targets_: unused
    :param num_units_to_plot: the number of units to plot
    :param seed: set the random seed for reproducibility.
    """

    # Randomly sample 'num_units_to_plot'.
    assert num_units_to_plot > 0
    num_units = winning_mask.shape[1]
    units = torch.randperm(num_units, generator=get_random_generator(seed))
    units = units[:num_units_to_plot]

    # Deduce winnings indices.
    winning_indices = winning_mask.max(dim=2).indices

    # Generate subplots.
    fig, axs = plt.subplots(1, num_units_to_plot, figsize=(6 * num_units_to_plot, 4))
    if num_units_to_plot == 1:
        axs = [axs]  # ensure this is subscriptable

    # Generate a plot for each unit.
    num_segments = winning_mask.shape[2]
    for i, unit in enumerate(units):
        indices = winning_indices[:, unit].cpu().numpy()
        plot_winning_segment_distribution(indices, num_segments, unit=unit, ax=axs[i])

    fig.tight_layout()
    return fig
