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

import math

import torch


def percent_active_dendrites(winning_mask, targets):
    """
    Returns a 3D torch tensor with shape (num_units, num_segments, num_categories)
    where cell i, j, c gives the fraction of inputs in category c for which segment j
    of unit i is active.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    with torch.no_grad():

        device = winning_mask.device

        # Assume the following:
        # - target values are zero-based
        # - the largest target value in the batch is that amongst all data
        num_categories = 1 + targets.max().item()
        _, num_units, num_segments = winning_mask.size()

        percent_active = torch.zeros((num_units, num_segments, 0))
        percent_active = percent_active.to(device)

        for t in range(num_categories):
            inds_t = torch.nonzero((targets == t).float(), as_tuple=True)
            num_examples_t = len(inds_t[0])

            percent_active_t = winning_mask[inds_t].sum(dim=0, dtype=torch.float)
            percent_active_t = percent_active_t / num_examples_t

            percent_active_t = percent_active_t.unsqueeze(2)
            percent_active = torch.cat((percent_active, percent_active_t), dim=2)

        return percent_active


def mean_selected_activations(dendrite_activations, winning_mask, targets):
    """
    Returns a 3D torch tensor with shape (num_units, num_segments, num_categories)
    where cell i, j, c gives the mean activation of the jth segment of unit i over all
    instances of category c for which segment j became active.

    :param dendrite_activations: 3D torch tensor with shape (batch_size, num_units,
                                 num_segments) in which entry b, i, j gives the
                                 activation of the ith unit's jth dendrite segment for
                                 example b
    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    with torch.no_grad():

        device = dendrite_activations.device

        # Assume the following:
        # - target values are zero-based
        # - the largest target value in the batch is that amongst all data
        num_categories = 1 + targets.max().item()
        _, num_units, num_segments = dendrite_activations.size()

        msa = torch.zeros((num_units, num_segments, 0))
        msa = msa.to(device)

        for t in range(num_categories):
            inds_t = torch.nonzero((targets == t).float(), as_tuple=True)

            num_selected_per_segment = winning_mask[inds_t].sum(dim=0)
            num_selected_per_segment[num_selected_per_segment == 0.0] = 1.0

            msa_t = dendrite_activations * winning_mask
            msa_t = msa_t[inds_t].sum(dim=0, dtype=torch.float)
            msa_t = msa_t / num_selected_per_segment

            msa_t = msa_t.unsqueeze(2)
            msa = torch.cat((msa, msa_t), dim=2)

        return msa


def dendrite_activations_by_unit(dendrite_activations, winning_mask, targets):
    """
    Returns a 2D torch tensor with shape (num_categories, num_units) where cell c, i
    gives the mean value (post-sigmoid) of the selected dendrite activation for unit i
    over all given examples from category c.

    :param dendrite_activations: 3D torch tensor with shape (batch_size, num_units,
                                 num_segments) in which entry b, i, j gives the
                                 activation of the ith unit's jth dendrite segment for
                                 example b
    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    with torch.no_grad():

        device = dendrite_activations.device

        # Assume the following:
        # - target values are zero-based
        # - the largest target value in the batch is that amongst all data
        num_categories = 1 + targets.max().item()
        _, num_units, _ = dendrite_activations.size()

        raw_winners = dendrite_activations * winning_mask

        selected_activations = torch.zeros((0, num_units))
        selected_activations = selected_activations.to(device)

        for t in range(num_categories):
            inds_t = torch.nonzero((targets == t).float(), as_tuple=True)

            # 'Select' dendrite activation for each example and each unit by summing
            # out segments; the sum on axis 2 only includes one non-zero entry
            selected_activations_t = raw_winners[inds_t]
            selected_activations_t = selected_activations_t.sum(dim=2)

            # Apply sigmoid and average across all examples
            selected_activations_t = torch.sigmoid(selected_activations_t)
            selected_activations_t = selected_activations_t.mean(dim=0)

            selected_activations_t = selected_activations_t.unsqueeze(0)
            selected_activations = torch.cat((selected_activations,
                                              selected_activations_t))

        return selected_activations


def hidden_activations_by_unit(activations, targets):
    """
    Returns a 2D torch tensor with shape (num_categories, num_units) where cell c, i
    gives the mean value (post-sigmoid) of the selected dendrite activation for unit i
    over all given examples from category c.

    :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                        b, i gives the activation of unit i for example b
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    with torch.no_grad():

        device = activations.device

        # Assume the following:
        # - target values are zero-based
        # - the largest target value in the batch is that amongst all data
        num_categories = 1 + targets.max().item()
        _, num_units = activations.size()

        # 'habu' is an abbreviation for 'hidden activations by unit'
        habu = torch.zeros((0, num_units))
        habu = habu.to(device)

        for t in range(num_categories):
            inds_t = torch.nonzero((targets == t).float(), as_tuple=True)
            habu_t = activations[inds_t]

            # Average activations across all examples with the same label
            habu_t = habu_t.mean(dim=0)

            habu_t = habu_t.unsqueeze(0)
            habu = torch.cat((habu, habu_t))

        return habu


def dendrite_overlap_matrix(winning_mask, targets):
    """ Returns a 3D torch tensor with shape (num_units, num_categories,
    num_categories) which represents num_units overlap matrices (one per unit) """
    with torch.no_grad():

        _, _, num_segments = winning_mask.size()
        percent_active = percent_active_dendrites(winning_mask, targets)

        # `percent_active` is an array with shape (num_units, num_segments,
        # num_categories); for each unit, compute the dot product between all pairs of
        # columns (where each column represents a categorical distribution over the
        # dendrite segments); the resulting tensor will have shape (num_units,
        # num_categories, num_categories)
        l2_norm = percent_active.norm(p=2, dim=1)
        l2_norm = l2_norm.unsqueeze(1).repeat((1, num_segments, 1))
        percent_active = percent_active / l2_norm

        overlap_matrix = torch.bmm(percent_active.transpose(1, 2), percent_active)
        return overlap_matrix


def dendrite_overlap(winning_mask, targets):
    """
    Returns a 1D torch tensor with shape (num_units,) where entry i gives the overlap
    score for the segments of unit i with respect to the specified categories. The
    overlap score for a single unit is computed via the following procedure:
        - for each class/category, compute the categorical distribution that gives the
          fraction of instances for which each segment becomes active (this function
          calls `percent_active_dendrites`),
        - normalize each probability vector to have L2 norm equal to 1,
        - compute all pairwise similarity scores (i.e., dot products) of dendrite
          activations between all classes/categories,
        - average the pairwise dot products.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    with torch.no_grad():

        device = winning_mask.device

        overlap_matrix = dendrite_overlap_matrix(winning_mask, targets)
        num_units, num_categories, _ = overlap_matrix.size()

        # The overlap score is simply the average of the off-diagonal entries of the
        # overlap matrix; in the ideal case with no dendrite overlap, the overlap
        # matrix is the identity matrix

        # Since the overlap matrix is symmetric, we only consider the lower half,
        # excluding the diagonal entries since they are all guaranteed to be 1

        # Mask for a lower triangular matrix
        ltril_mask = torch.ones((num_categories, num_categories))
        ltril_mask = ltril_mask.to(device)
        ltril_mask = ltril_mask.tril(diagonal=-1)
        ltril_mask = ltril_mask.unsqueeze(0).repeat((num_units, 1, 1))

        overlap_score = (overlap_matrix * ltril_mask).sum(dim=(1, 2))
        overlap_score /= (0.5 * num_categories * (num_categories - 1))
        return overlap_score


def dendrite_duty_cycle(winning_mask):
    """
    Returns a 2D torch tensor with shape (number of units, number of segments) where
    entry i, j gives the duty cycle of segment j in the set of dendrite segments for
    unit i. The duty cycle for each segment is computed offline.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    """
    with torch.no_grad():

        num_examples, _, _ = winning_mask.size()
        return winning_mask.sum(dim=0, dtype=torch.float) / num_examples


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
