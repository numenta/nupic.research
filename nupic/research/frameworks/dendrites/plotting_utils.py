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
import numpy as np
import torch

from .utils import (
    dendrite_duty_cycle,
    dendrite_overlap,
    dendrite_overlap_matrix,
    entropy,
    mean_selected_activations,
    percent_active_dendrites,
)


def plot_dendrite_activations(
    dendrite_segments,
    context_vectors,
    selection_criterion,
    mask_values=None,
    unit_to_plot=0
):
    """
    Returns a heatmap of dendrite activations (given dendrite weights for a single
    neuron and context vectors) plotted using matplotlib.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: 2D torch tensor with shape (num_contexts, dim_context)
    :param selection_criterion: either "regular" or "absolute"; "regular" annotates the
                                maxmimum activation per context, whereas "absolute"
                                annotates the absolute maximum activation per context
    :param mask_values: list of the routing function's mask values for all contexts,
                        for output unit `unit_to_plot`; unused if None
    :param unit_to_plot: index of the unit for which to plot dendrite activations;
                         plots activations of unit 0 by default
    """
    with torch.no_grad():

        assert selection_criterion in ("regular", "absolute")

        num_contexts = context_vectors.size(0)
        num_units, num_dendrites, _ = dendrite_segments.weights.size()

        assert 0 <= unit_to_plot < num_units

        # Compute activation values
        activations = dendrite_segments(context_vectors)
        activations = activations[:, unit_to_plot, :].T

        x_labels = ["context {}".format(j) for j in range(num_contexts)]
        if mask_values is not None:
            assert len(mask_values) == num_contexts
            x_labels = ["{} [{}]".format(label, mask_values[j])
                        for j, label in enumerate(x_labels)]
        y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

        # Find the range of activation values to anchor the colorbar
        vmax = activations.abs().max().item()
        vmin = -1.0 * vmax

        # Use matplotlib to plot the activation heatmap
        plt.cla()
        fig, ax = plt.subplots()
        ax.imshow(activations.detach().cpu().numpy(), cmap="coolwarm_r", vmin=vmin,
                  vmax=vmax)

        ax.set_xticks(np.arange(num_contexts))
        ax.set_yticks(np.arange(num_dendrites))

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()

        # Annotate just the maximum or absolute maximum activation for each context
        top_activation_dendrite_per_context = torch.argmax(
            activations.abs() if selection_criterion == "absolute" else activations,
            axis=0
        )
        for j, i in enumerate(top_activation_dendrite_per_context):
            val = round(activations[i, j].item(), 2)
            ax.text(j, i, val, ha="center", va="center", color="w")

        figure = plt.gcf()
        return figure


def plot_percent_active_dendrites(
    dendrite_segments,
    context_vectors,
    selection_criterion,
    category_names=None,
    unit_to_plot=0
):
    """
    Returns a heatmap with shape (number of dendrites, number of categories) where cell
    i, j in the heatmap gives the percentage of inputs in category j for which dendrite
    i is active (for a single unit).

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param category_names: list of category names to label each column of the heatmap,
                           and needs to align with category order in `context_vectors`;
                           unused if None
    :param unit_to_plot: index of the unit for which to plot percent active dendrites;
                         plots unit 0 by default
    """
    assert selection_criterion in ("regular", "absolute")
    if category_names is not None:
        assert len(context_vectors) == len(category_names)

    num_categories = len(context_vectors)
    _, num_dendrites, _ = dendrite_segments.weights.size()

    x_labels = ["category {}".format(j) for j in range(len(context_vectors))]
    if category_names is not None:
        x_labels = category_names
    y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

    percentage_activations = percent_active_dendrites(
        dendrite_segments=dendrite_segments,
        context_vectors=context_vectors,
        selection_criterion=selection_criterion
    )
    percentage_activations = percentage_activations[unit_to_plot, :, :]
    percentage_activations = percentage_activations.detach().cpu().numpy()

    # Find the maximum percentage activation value to anchor the colorbar, and use
    # matplotlib to plot the heatmap
    vmax = np.max(percentage_activations)

    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(percentage_activations, cmap="copper", vmin=0.0, vmax=vmax)

    ax.set_xticks(np.arange(num_categories))
    ax.set_yticks(np.arange(num_dendrites))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    # Annotate all percentage activations
    for i in range(percentage_activations.shape[0]):
        for j in range(percentage_activations.shape[1]):
            val = np.round(percentage_activations[i, j], 2)
            ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure


def plot_mean_selected_activations(
    dendrite_segments,
    context_vectors,
    selection_criterion,
    category_names=None,
    unit_to_plot=0
):
    """
    Returns a 3D torch tensor with shape (num_units, num_dendrites, num_categories)
    where cell k, i, j gives the mean activation of the ith dendrite of unit k over all
    instances of category j for which dendrite i became active.

    Returns a heatmap with shape (number of dendrites, number of categories) where cell
    i, j in the heatmap gives the mean activation of the dendrite i over all instances
    of category j for which dendrite i became active. As there are multiple dendrite
    segments, the heatmap is created for the specified unit.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param category_names: list of category names to label each column of the heatmap,
                           and needs to align with category order in `context_vectors`;
                           unused if None
    :param unit_to_plot: index of the unit for which to plot mean selected activations;
                         plots unit 0 by default
    """
    assert selection_criterion in ("regular", "absolute")
    if category_names is not None:
        assert len(context_vectors) == len(category_names)

    num_categories = len(context_vectors)
    num_units, num_dendrites, _ = dendrite_segments.weights.size()

    assert 0 <= unit_to_plot < num_units

    x_labels = ["category {}".format(j) for j in range(len(context_vectors))]
    if category_names is not None:
        x_labels = category_names
    y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

    msa = mean_selected_activations(dendrite_segments, context_vectors,
                                    selection_criterion)
    msa = msa[unit_to_plot, :, :]
    msa = msa.detach().cpu().numpy()

    # Find the largest absolute mean selected activation value to anchor the colorbar,
    # and use matplotlib to plot the heatmap
    vmax = np.nanmax(np.abs(msa))
    vmin = -vmax

    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(msa, cmap="coolwarm_r", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(num_categories))
    ax.set_yticks(np.arange(num_dendrites))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    # Annotate all mean selected activations
    for i in range(msa.shape[0]):
        for j in range(msa.shape[1]):
            val = round(msa[i, j], 2)
            ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure


def plot_dendrite_overlap_matrix(
    dendrite_segments,
    context_vectors,
    selection_criterion,
    category_names=None,
    unit_to_plot=0
):
    """
    Returns a heatmap with shape (number of categories, number of categories) where
    cell i, j gives the overlap in dendrite activations between categories i and j for
    the dendrite segments of the specified unit. The value in each cell can be
    interpreted as a similarity measure in dendrite activations between categories i
    and j; if the exact same dendrites are active for the same fraction of instances
    across both categories, the dendrite overlap is 1; if any dendrite that is active
    for category i and inactive for category j (and vice-versa), the dendrite overlap
    is 0. The resulting heatmap is symmetric.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param category_names: list of category names to label each column of the heatmap,
                           and needs to align with category order in `context_vectors`;
                           unused if None
    :param unit_to_plot: index of the unit for which to plot the overlap matrix; plots
                         unit 0 by default
    """
    assert selection_criterion in ("regular", "absolute")
    if category_names is not None:
        assert len(context_vectors) == len(category_names)

    num_categories = len(context_vectors)

    labels = ["category {}".format(j) for j in range(len(context_vectors))]
    if category_names is not None:
        labels = category_names

    overlap_matrix = dendrite_overlap_matrix(
        dendrite_segments=dendrite_segments,
        context_vectors=context_vectors,
        selection_criterion=selection_criterion
    )
    overlap_matrix = overlap_matrix[unit_to_plot, :, :]
    overlap_matrix = overlap_matrix.detach().cpu().numpy()

    # `overlap_matrix` is symmetric, hence we can set all values above the main
    # diagonal to np.NaN so they don't appear in the visualization
    for i in range(num_categories):
        for j in range(i + 1, num_categories):
            overlap_matrix[i, j] = np.nan

    # Anchor the colorbar to the range [0, 1]
    # vmax = np.max(overlap_matrix)
    plt.cla()
    fig, ax = plt.subplots()
    ax.imshow(overlap_matrix, cmap="OrRd", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(num_categories))
    ax.set_yticks(np.arange(num_categories))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    # Annotate all overlap values
    for i in range(num_categories):
        for j in range(i + 1):
            val = np.round(overlap_matrix[i, j].item(), 2)
            ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure


def plot_overlap_scores_distribution(dendrite_segments, context_vectors,
                                     selection_criterion):
    """
    Returns a histogram which gives the distribution of dendrite overlap scores for all
    units over the examples specified by the context vectors. Each data point in the
    histogram is the overlap score corresponding to the dendrite segments of a single
    unit. See `dendrite_overlap` for more details.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    overlap_scores = dendrite_overlap(dendrite_segments, context_vectors,
                                      selection_criterion)

    plt.cla()
    plt.hist(x=overlap_scores.tolist(), bins=np.arange(0.0, 1.0, 0.05),
             color="m", edgecolor="k")

    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlabel("Overlap score")
    plt.ylabel("Segment frequency")
    plt.xlim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()

    figure = plt.gcf()
    return figure


def plot_entropy_distribution(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a histogram which gives the distribution of entropy values of dendrite
    segments over the examples specified by the context vectors. Each data point in the
    histogram is the observed entropy of a set of dendrite segments corresponding to a
    single unit. The entropy is the computed using the empirical distribution of the
    fraction of instances for which each dendrite became active.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: a single 2D torch tensor of context vectors across multiple
                            classes, or iterable of 2D torch tensors with shape
                            (num_examples, dim_context) where each 2D tensor gives a
                            batch of context vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    num_units, _, _ = dendrite_segments.weights.size()
    duty_cycle = dendrite_duty_cycle(dendrite_segments, context_vectors,
                                     selection_criterion)

    entropies = [entropy(duty_cycle[unit, :]) for unit in range(num_units)]
    max_entropy = entropies[0][1]
    entropies = [ent[0] for ent in entropies]

    plt.cla()
    plt.hist(x=entropies, bins=np.arange(0.0, max_entropy, 0.1), color="g",
             edgecolor="k")

    plt.xticks(np.arange(0.0, 1.0, 0.2))
    plt.xlabel("Entropy  (max entropy: {})".format(round(max_entropy, 2)))
    plt.ylabel("Segment frequency")
    plt.xlim(0.0, max_entropy)
    plt.grid(True)
    plt.tight_layout()

    figure = plt.gcf()
    return figure
