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

import math

import numpy as np
import torch


def percent_active_dendrites(dendrite_weights, context_vectors, selection_criterion):
    """
    Returns a 2D NumPy array with shape (number of dendrites, number of categories)
    where cell i, j gives the fraction of inputs in category j for which dendrite i is
    active (for a single unit). The columns in the returned array sum to 1.

    :param dendrite_weights: 2D torch tensor with shape (num_dendrites, dim_context);
                             note these weights are specific to a single unit
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    num_dendrites = dendrite_weights.size(0)

    percentage_activations = np.zeros((num_dendrites, 0))
    for j in range(len(context_vectors)):
        num_examples = context_vectors[j].size(0)

        activations = torch.matmul(dendrite_weights, context_vectors[j].T)
        activations = activations.detach().cpu().numpy()
        if selection_criterion == "absolute":
            activations = np.abs(activations)

        selected = 1.0 * (np.max(activations, axis=0) == activations)
        selected = np.sum(selected, axis=1) / num_examples

        selected = selected.reshape(-1, 1)
        percentage_activations = np.concatenate(
            (percentage_activations, selected),
            axis=1
        )

    return percentage_activations


def dendrite_overlap_matrix(dendrite_weights, context_vectors, selection_criterion):
    percentage_activations = percent_active_dendrites(
        dendrite_weights=dendrite_weights,
        context_vectors=context_vectors,
        selection_criterion=selection_criterion
    )

    # `percentage_activations` is an array with shape (num_dendrites, num_categories);
    # compute the dot product between all pairs of columns into a matrix with shape
    # (num_categories, num_categories)
    percentage_activations /= np.linalg.norm(percentage_activations, axis=0)
    overlap_matrix = np.matmul(percentage_activations.T, percentage_activations)
    return overlap_matrix


def dendrite_overlap(dendrite_weights, context_vectors, selection_criterion):
    """
    Returns the dendrite overlap score of a given set of dendrites (specific to a
    single output unit) in relation to specified set of classes/categories, which is
    computed via the following procedure:
        - for each class/category, compute the categorical distribution that gives the
          fraction of instances for which each dendrite becomes active (this function
          calls `percent_active_dendrites`),
        - normalize each probability vector to have L2 norm equal to 1,
        - compute all pairwise similarity scores (i.e., dot products) of dendrite
          activations between all classes/categories,
        - average the pairwise dot products.

    :param dendrite_weights: 2D torch tensor with shape (num_dendrites, dim_context);
                             note these weights are specific to a single unit
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    overlap_matrix = dendrite_overlap_matrix(
        dendrite_weights, context_vectors, selection_criterion
    )

    num_categories = overlap_matrix.shape[0]

    # The overlap score is simply the average of the off-diagonal entries of the
    # overlap matrix; in the ideal case with no dendrite overlap, the overlap matrix is
    # the identity matrix

    # Since the overlap matrix is symmetric, we only consider the lower half, excluding
    # the diagonal entries since they are all guaranteed to be 1
    overlap_score = np.tril(overlap_matrix, k=-1).sum()
    overlap_score /= (0.5 * num_categories * (num_categories - 1))
    return overlap_score


def dendrite_duty_cycle(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a 2D torch tensor with shape (number of units, number of dendrites) where
    entry i, j gives the duty cycle of dendrite j in the set of dendrite segments for
    unit i. The duty cycle for each dendrite is computed offline.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: a single 2D torch tensor of context vectors across multiple
                            classes, or iterable of 2D torch tensors with shape
                            (num_examples, dim_context) where each 2D tensor gives a
                            batch of context vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    if not isinstance(context_vectors, torch.Tensor):
        context_vectors = torch.cat(context_vectors)
    num_examples = context_vectors.size(0)

    activations = dendrite_segments(context_vectors)
    if selection_criterion == "absolute":
        activations = activations.abs()

    duty_cycle = (activations.max(axis=2, keepdims=True).values == activations)
    duty_cycle = duty_cycle.sum(axis=0, dtype=torch.float)
    duty_cycle = duty_cycle / num_examples

    return duty_cycle


def online_dendrite_duty_cycle(dendrite_segments, context_vectors, selection_criterion,
                               online=False, batch_size=32, period=1000):
    """
    Returns a 2D torch tensor with shape (number of units, number of dendrites) where
    entry i, j gives the duty cycle of dendrite j in the set of dendrite segments for
    unit i, computed using the online formula for duty cycles. See Equation 8 in the
    spatial pooling paper for more details:

    https://www.biorxiv.org/content/10.1101/085035v2

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: a single 2D torch tensor of context vectors across multiple
                            classes, or iterable of 2D torch tensors with shape
                            (num_examples, dim_context) where each 2D tensor gives a
                            batch of context vectors from the same category
    :param selection_criterion: the criterion for selecting which dendrites become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    :param batch_size: the batch size used to calculate duty cycles; only relevant if
                       `online` is `True`
    :param period: the period used to calculate duty cycles; only relevant if `online`
                   is `True`
    """
    if not isinstance(context_vectors, torch.Tensor):
        context_vectors = torch.cat(context_vectors)
    num_examples = context_vectors.size(0)

    batch_start = 0
    batch_end = min(num_examples, batch_size)

    duty_cycle = torch.zeros((dendrite_segments.num_units,
                              dendrite_segments.num_segments))

    while batch_start < num_examples:
        batch = context_vectors[batch_start:batch_end, :]
        activations = dendrite_segments(batch)
        if selection_criterion == "absolute":
            activations = activations.abs()

        num_activations = activations.max(axis=2, keepdims=True).values == activations
        num_activations = num_activations.sum(axis=0, dtype=torch.float)

        # Update the duty cycle using the formula:
        # new_duty_cycle = ((period - batch_size) / period) * old_duty_cycle
        #                  + (1 / period) * batch_active_count
        duty_cycle = duty_cycle * (period - (batch_end - batch_start))
        duty_cycle = duty_cycle + num_activations
        duty_cycle = duty_cycle / period

        batch_start = batch_start + batch_size
        batch_end = min(num_examples, batch_start + batch_size)

    return duty_cycle


def entropy(x):
    """
    Returns a tuple of scalars (entropy value, maximum possible entropy value) which
    gives the entropy of dendrite segments and the maximmum entropy that can be
    achieved, respectively.

    :param x: a 1D torch tensor representing a probability distribution where entry i
              gives the number of times dendrite i became active
    """
    num_dendrites = x.shape[0]
    if not x.sum().item() == 1.0:
        x = x / x.sum()

    _entropy = -(x * x.log())
    _entropy[x == 0.0] = 0.0
    _entropy = _entropy.sum().item()
    _max_entropy = math.log(num_dendrites)

    return _entropy, _max_entropy
