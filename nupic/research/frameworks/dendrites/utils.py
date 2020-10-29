# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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


def plot_dendritic_activations(
    dendritic_weights,
    context_vectors,
    mask_values,
    use_absolute_activations=False
):
    """
    Returns a heatmap of dendrite activations (given dendritic weights for a single
    neuron and context vectors) plotted using matplotlib. Note that the user must be
    logged in to wandb on both browser and terminal to view the resulting plots, and
    this can be done via the following command:

    $ wandb login your-login-key

    :param dendritic_weights: 2D torch tensor with shape (num_dendrites, dim_context)
    :param context_vectors: 2D torch tensor with shape (num_contexts, dim_context)
    :param mask_values: list of the routing function's mask values for the output unit
                        corresponding to `dendritic_weights`, across all contexts
    :param use_absolute_activations: plots absolute activation values if True
    """
    assert dendritic_weights.size(1) == context_vectors.size(1)

    plt.cla()

    activations = torch.matmul(dendritic_weights, context_vectors.T)
    if use_absolute_activations:
        activations = torch.abs(activations)
    activations = activations.detach().cpu().numpy()

    num_contexts = context_vectors.size(0)
    num_dendrites = dendritic_weights.size(0)
    assert len(mask_values) == num_contexts

    x_labels = [
        "context {} [{}]".format(j, mask_values[j]) for j in range(num_contexts)
    ]
    y_labels = ["dendrite {}".format(j) for j in range(num_dendrites)]

    # Use matplotlib to plot the activation heatmap
    fig, ax = plt.subplots()
    ax.imshow(activations, cmap="coolwarm_r")

    ax.set_xticks(np.arange(num_contexts))
    ax.set_yticks(np.arange(num_dendrites))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()

    # Annotate just the top absolute activation for each context
    top_activation_dendrite_per_context = np.argmax(np.abs(activations), axis=0)
    for j, i in enumerate(top_activation_dendrite_per_context):
        val = np.round(activations[i, j], 2)
        ax.text(j, i, val, ha="center", va="center", color="w")

    figure = plt.gcf()
    return figure
