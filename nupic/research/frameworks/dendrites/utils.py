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

import torch


def percent_active_dendrites(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a 3D torch tensor with shape (num_units, num_segments, num_categories)
    where cell k, i, j gives the fraction of inputs in category j for which segment i
    of unit k is active.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which segments become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    with torch.no_grad():

        num_units, num_segments, _ = dendrite_segments.weights.size()

        percentage_activations = torch.zeros((num_units, num_segments, 0))
        for j in range(len(context_vectors)):
            num_examples = context_vectors[j].size(0)

            activations = dendrite_segments(context_vectors[j])
            if selection_criterion == "absolute":
                activations = activations.abs()

            selected = (activations.max(axis=2, keepdims=True).values == activations)
            selected = selected.sum(axis=0, dtype=torch.float) / num_examples

            selected = selected.unsqueeze(2)
            percentage_activations = torch.cat((percentage_activations, selected),
                                               dim=2)

        return percentage_activations


def mean_selected_activations(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a 3D torch tensor with shape (num_units, num_segments, num_categories)
    where cell k, i, j gives the mean activation of the ith segment of unit k over all
    instances of category j for which segment i became active.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which segments become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    with torch.no_grad():

        num_units, num_segments, _ = dendrite_segments.weights.size()

        mean_selected_activations = torch.zeros((num_units, num_segments, 0))
        for j in range(len(context_vectors)):
            activations = dendrite_segments(context_vectors[j])

            selected = activations
            if selection_criterion == "absolute":
                selected = activations.abs()
            selected = (selected.max(axis=2, keepdims=True).values == selected)

            num_selected_per_segment = selected.sum(axis=0)
            num_selected_per_segment[num_selected_per_segment == 0.0] = 1.0

            msa_j = activations * selected
            msa_j = msa_j.sum(axis=0, dtype=torch.float) / num_selected_per_segment

            msa_j = msa_j.unsqueeze(2)
            mean_selected_activations = torch.cat((mean_selected_activations, msa_j),
                                                  dim=2)

        return mean_selected_activations


def dendrite_overlap_matrix(dendrite_segments, context_vectors, selection_criterion):
    """ Returns a 3D torch tensor with shape (num_units, num_categories,
    num_categories) which represents num_units overlap matrices (one per unit) """
    with torch.no_grad():

        _, num_segments, _ = dendrite_segments.weights.size()

        percentage_activations = percent_active_dendrites(
            dendrite_segments=dendrite_segments,
            context_vectors=context_vectors,
            selection_criterion=selection_criterion
        )

        # `percentage_activations` is an array with shape (num_units, num_segments,
        # num_categories); for each unit's dendrite segments, compute the dot product
        # between all pairs of columns; the resulting tensor will have shape
        # (num_units, num_categories, num_categories)
        l2_norm = percentage_activations.norm(p=2, dim=1)
        l2_norm = l2_norm.unsqueeze(1).repeat((1, num_segments, 1))
        percentage_activations /= l2_norm

        overlap_matrix = torch.bmm(percentage_activations.transpose(1, 2),
                                   percentage_activations)
        return overlap_matrix


def dendrite_overlap(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a 1D torch tensor with shape (num_units,) where entry k gives the overlap
    score for the segments of unit k in relation to the specified context veectors and
    specified classes/categories. The overlap score for a single unit is computed via
    the following procedure:
        - for each class/category, compute the categorical distribution that gives the
          fraction of instances for which each segment becomes active (this function
          calls `percent_active_dendrites`),
        - normalize each probability vector to have L2 norm equal to 1,
        - compute all pairwise similarity scores (i.e., dot products) of dendrite
          activations between all classes/categories,
        - average the pairwise dot products.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: iterable of 2D torch tensors with shape (num_examples,
                            dim_context) where each 2D tensor gives a batch of context
                            vectors from the same category
    :param selection_criterion: the criterion for selecting which segments become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    with torch.no_grad():

        overlap_matrix = dendrite_overlap_matrix(dendrite_segments, context_vectors,
                                                 selection_criterion)

        num_units, num_categories, _ = overlap_matrix.size()

        # The overlap score is simply the average of the off-diagonal entries of the
        # overlap matrix; in the ideal case with no dendrite overlap, the overlap
        # matrix is the identity matrix

        # Since the overlap matrix is symmetric, we only consider the lower half,
        # excluding the diagonal entries since they are all guaranteed to be 1

        # Mask for a lower triangular matrix
        ltril_mask = torch.ones((num_categories, num_categories))
        ltril_mask = ltril_mask.tril(diagonal=-1)
        ltril_mask = ltril_mask.unsqueeze(0).repeat((num_units, 1, 1))

        overlap_score = (overlap_matrix * ltril_mask).sum(dim=(1, 2))
        overlap_score /= (0.5 * num_categories * (num_categories - 1))
        return overlap_score


def dendrite_duty_cycle(dendrite_segments, context_vectors, selection_criterion):
    """
    Returns a 2D torch tensor with shape (number of units, number of segments) where
    entry i, j gives the duty cycle of segment j in the set of dendrite segments for
    unit i. The duty cycle for each segment is computed offline.

    :param dendrite_segments: `DendriteSegments` object
    :param context_vectors: a single 2D torch tensor of context vectors across multiple
                            classes, or iterable of 2D torch tensors with shape
                            (num_examples, dim_context) where each 2D tensor gives a
                            batch of context vectors from the same category
    :param selection_criterion: the criterion for selecting which segments become
                                active; either "regular" (for `GatingDendriticLayer`)
                                or "absolute" (for `AbsoluteMaxGatingDendriticLayer`)
    """
    with torch.no_grad():

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


def entropy(x):
    """
    Returns a tuple of scalars (entropy value, maximum possible entropy value) which
    gives the entropy of dendrite segments and the maximum entropy that can be
    achieved, respectively.

    :param x: a 1D torch tensor representing a probability distribution where entry i
              gives the number of times segment i became active
    """
    num_segments = x.shape[0]
    if not x.sum().item() == 1.0:
        x = x / x.sum()

    _entropy = -(x * x.log())
    _entropy[x == 0.0] = 0.0
    _entropy = _entropy.sum().item()
    _max_entropy = math.log(num_segments)

    return _entropy, _max_entropy
