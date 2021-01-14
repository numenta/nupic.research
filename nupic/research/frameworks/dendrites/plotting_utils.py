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
    dendrite_overlap_matrix,
    entropy,
    percent_active_dendrites,
)


def plot_dendrite_activations(
    dendrite_weights,
    context_vectors,
    annotation_type,
    mask_values=None
):
    """
    Returns a heatmap of dendrite activations (given dendrite weights for a single
    neuron and context vectors) plotted using matplotlib. Note that the user must be
    logged in to wandb on both browser and terminal to view the resulting plots, and
    this can be done via the following command:

    $ wandb login your-login-key

    :param dendrite_weights: 2D torch tensor with shape (num_dendrites, dim_context)
    :param context_vectors: 2D torch tensor with shape (num_contexts, dim_context)
    :param annotation_type: either "regular" or "absolute"; "regular" annotates the
                            maxmimum activation per context, whereas "absolute"
                            annotates the absolute maximum activation per context
    :param mask_values: list of the routing function's mask values for the output unit
                        corresponding to `dendrite_weights` across all contexts;
                        unused if None
    """
    assert dendrite_weights.size(1) == context_vectors.size(1)
    assert annotation_type in ("regular", "absolute")

    plt.cla()

    activations = torch.matmul(dendrite_weights, context_vectors.T)
    activations = activations.detach().cpu().numpy()

    num_contexts = context_vectors.size(0)
    num_dendrites = dendrite_weights.size(0)

    x_labels = ["context {}".format(j) for j in range(num_contexts)]
    if mask_values is not None:
        assert len(mask_values) == num_contexts
        x_labels = [
            "{} [{}]".format(label, mask_values[j]) for j, label in enumerate(x_labels)
        ]
    y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

    # Find the range of activation values to anchor the colorbar
    vmax = np.abs(activations).max()
    vmin = -1.0 * vmax

    # Use matplotlib to plot the activation heatmap
    fig, ax = plt.subplots()
    ax.imshow(activations, cmap="coolwarm_r", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(num_contexts))
    ax.set_yticks(np.arange(num_dendrites))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    # Annotate just the maximum or absolute maximum activation for each context
    top_activation_dendrite_per_context = np.argmax(
        np.abs(activations) if annotation_type == "absolute" else activations,
        axis=0
    )
    for j, i in enumerate(top_activation_dendrite_per_context):
        val = np.round(activations[i, j], 2)
        ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure


def plot_percent_active_dendrites(
    dendrite_weights,
    context_vectors,
    selection_criterion,
    category_names=None
):
    """
    Returns a heatmap with shape (number of dendrites, number of categories) where cell
    i, j in the heatmap gives the percentage of inputs in category j for which dendrite
    i is active (for a single unit). Note that the user must be logged in to wandb on
    both browser and terminal to view the resulting plots, and this can be done via the
    following command:

    $ wandb login your-login-key

    :param dendrite_weights: 2D torch tensor with shape (num_dendrites, dim_context)
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param category_names: list of category names to label each column of the heatmap,
                           and needs to align with category order in `context_vectors`;
                           unused if None
    """
    assert all(
        [dendrite_weights.size(1) == batch.size(1) for batch in context_vectors]
    )
    assert selection_criterion in ("regular", "absolute")
    if category_names is not None:
        assert len(context_vectors) == len(category_names)

    num_categories = len(context_vectors)
    num_dendrites = dendrite_weights.size(0)

    x_labels = ["category {}".format(j) for j in range(len(context_vectors))]
    if category_names is not None:
        x_labels = category_names
    y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

    percentage_activations = percent_active_dendrites(
        dendrite_weights=dendrite_weights,
        context_vectors=context_vectors,
        selection_criterion=selection_criterion
    )

    plt.cla()

    # Find the maximum percentage activation value to anchor the colorbar, and use
    # matplotlib to plot the heatmap
    vmax = np.max(percentage_activations)
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
    dendrite_weights,
    context_vectors,
    selection_criterion,
    category_names=None
):
    """
    Returns a heatmap with shape (number of dendrites, number of categories) where cell
    i, j in the heatmap gives the mean selected activation of dendrite i under category
    j. That is, for all context inputs in category j such that dendrite i was selected,
    the cell value gives the mean activation. Note that the user must be logged in to
    wandb on both browser and terminal to view the resulting plots, and this can be
    done via the following command:

    $ wandb login your-login-key

    :param dendrite_weights: 2D torch tensor with shape (num_dendrites, dim_context)
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param category_names: list of category names to label each column of the heatmap,
                           and needs to align with category order in `context_vectors`;
                           unused if None
    """
    assert all(
        [dendrite_weights.size(1) == batch.size(1) for batch in context_vectors]
    )
    assert selection_criterion in ("regular", "absolute")
    if category_names is not None:
        assert len(context_vectors) == len(category_names)

    plt.cla()

    num_categories = len(context_vectors)
    num_dendrites = dendrite_weights.size(0)

    x_labels = ["category {}".format(j) for j in range(len(context_vectors))]
    if category_names is not None:
        x_labels = category_names
    y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

    mean_selected_activations = np.zeros((num_dendrites, 0))
    for j in range(len(context_vectors)):
        activations = torch.matmul(dendrite_weights, context_vectors[j].T)
        activations = activations.detach().cpu().numpy()

        selected = activations
        if selection_criterion == "absolute":
            selected = np.abs(selected)
        selected = 1.0 * (np.max(selected, axis=0) == selected)

        num_selected_per_dendrite = np.sum(selected, axis=1)
        np.place(num_selected_per_dendrite, num_selected_per_dendrite == 0.0, 1.0)

        selected = activations * selected
        selected = np.sum(selected, axis=1) / num_selected_per_dendrite

        selected = selected.reshape(-1, 1)
        mean_selected_activations = np.concatenate(
            (mean_selected_activations, selected),
            axis=1
        )

    assert mean_selected_activations.shape[0] == num_dendrites
    assert mean_selected_activations.shape[1] == num_categories

    # Find the largest absolute mean selected activation value to anchor the colorbar,
    # and use matplotlib to plot the heatmap
    vmax = np.nanmax(np.abs(mean_selected_activations))
    print(vmax)
    vmin = -vmax
    fig, ax = plt.subplots()
    ax.imshow(mean_selected_activations, cmap="coolwarm_r", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(num_categories))
    ax.set_yticks(np.arange(num_dendrites))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    # Annotate all mean selected activations
    for i in range(mean_selected_activations.shape[0]):
        for j in range(mean_selected_activations.shape[1]):
            val = np.round(mean_selected_activations[i, j], 2)
            ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure


def plot_dendrite_overlap_matrix(
    dendrite_weights,
    context_vectors,
    selection_criterion,
    category_names=None
):
    """
    Returns a heatmap with shape (number of categories, number of categories) where
    cell i, j gives the overlap in dendrite activations between categories i and j. The
    value in each cell can be interpreted as a similarity measure in dendrite
    activations between categories i and j; if the exact same dendrites are active for
    the same fraction of instances across both categories, the dendrite overlap is 1;
    if any dendrite that is active for category i and inactive for category j (and
    vice-versa), the dendrite overlap is 0. The resulting heatmap is symmetric. Note
    that the user must be logged in to wandb on both browser and terminal to view the
    resulting plots, and this can be done via the following command:

    $ wandb login your-login-key

    :param dendrite_weights: 2D torch tensor with shape (num_dendrites, dim_context)
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param category_names: list of category names to label each column of the heatmap,
                           and needs to align with category order in `context_vectors`;
                           unused if None
    """
    assert all(
        [dendrite_weights.size(1) == batch.size(1) for batch in context_vectors]
    )
    assert selection_criterion in ("regular", "absolute")
    if category_names is not None:
        assert len(context_vectors) == len(category_names)

    num_categories = len(context_vectors)

    labels = ["category {}".format(j) for j in range(len(context_vectors))]
    if category_names is not None:
        labels = category_names

    overlap_matrix = dendrite_overlap_matrix(
        dendrite_weights=dendrite_weights,
        context_vectors=context_vectors,
        selection_criterion=selection_criterion
    )

    plt.cla()

    # `overlap_matrix` is symmetric, hence we can set all values above the main
    # diagonal to np.NaN so they don't appear in the visualization
    for i in range(num_categories):
        for j in range(i + 1, num_categories):
            overlap_matrix[i, j] = np.nan

    # Anchor the colorbar to the range [0, 1]
    # vmax = np.max(overlap_matrix)
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
            val = np.round(overlap_matrix[i, j], 2)
            ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure


def plot_entropy_distribution(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a histogram which gives the distribution of entropy values of dendrite
    segments over the examples specified by the context vectors. Each single data point
    in the histogram is the observed entropy of a set of dendrite segments
    corresponding to a single unit. The entropy is the computed using the empirical
    distribution of the fraction of instances for which each dendrite became active.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: a single 2D torch tensor of context vectors across multiple
                            classes, or iterable of 2D torch tensors with shape
                            (num_examples, dim_context) where each 2D tensor gives a
                            batch of context vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    duty_cycle = dendrite_duty_cycle(dendrite_segments, context_vectors,
                                     selection_criterion)

    num_units = duty_cycle.shape[0]
    entropies = [entropy(duty_cycle[unit, :]) for unit in range(num_units)]
    max_entropy = entropies[0][1]
    entropies = [ent[0] for ent in entropies]

    plt.cla()
    plt.hist(x=entropies, bins=np.arange(0.0, max_entropy, 0.1), color="g",
             edgecolor="k")

    plt.xlabel("entropy  (max entropy: {})".format(round(max_entropy, 2)))
    plt.ylabel("Segment frequency")
    plt.xlim(0.0, max_entropy)
    plt.grid(True)
    plt.tight_layout()

    figure = plt.gcf()
    return figure
