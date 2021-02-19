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

import numpy as np
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
    gives the mean value of hidden activations for unit i over all given examples from
    category c.

    NOTE: This function is not specific to dendrites, and can be used with modules that
    both include and don't include dendrite segments.

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


def winning_segment_indices(winning_mask, units=None):
    """
    Returns the indices of the winning segments for the given units.

    :param winning_mask: 3D torch tensor with shape (batch_size, num_units,
                         num_segments) in which entry b, i, j is 1 iff the ith unit's
                         jth dendrite segment won for example b, 0 otherwise
    :param units: (optional) a list of units; winning indices will only be returned
                  for these units; default to all units.
    """
    units = units or ...  # the ellipses is equivalent to all units
    winning_indices = winning_mask.max(dim=2).indices  # shape batch_size x num_units
    return winning_indices[:, units]


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


def repr_overlap_matrix(activations, targets):
    """
    Returns a 2D torch tensor with shape (num_categories, num_categories) where cell
    c1, c2 gives the mean value of pairwise representation overlap across all pairs of
    examples between classes c1 and c2. Each individual pairwise representation overlap
    is simply the fraction of hidden units that are active in both examples.

    NOTE: This function is not specific to dendrites, and can be used with modules that
    both include and don't include dendrite segments.

    :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                        b, i gives the activation of unit i for example b
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    with torch.no_grad():

        repr_overlaps = repr_overlap_stats(activations, targets)

        # Assume the following:
        # - target values are zero-based
        # - the largest target value in the batch is that amongst all data
        num_categories = 1 + targets.max().item()

        overlap_matrix = torch.zeros((num_categories, num_categories))

        # For each pair of categories, compute the mean value of pairwise
        # representation overlaps using all pairs of the form (x1, x2) where x1 belongs
        # to category c1, and x2 belongs to category c2
        for t1 in range(num_categories):
            for t2 in range(1 + t1):

                mean_overlap_val = np.mean(repr_overlaps[t1][t2])
                overlap_matrix[t1, t2] = mean_overlap_val

        return overlap_matrix


def repr_overlap_distributions(activations, targets):
    """
    Returns (list of floats, list of floats) where the first list gives representation
    overlap values between all pairs of samples in different classes, and the second
    list gives representation overlaps values between all pairs of samples in the same
    class.

    NOTE: This function is not specific to dendrites, and can be used with modules that
    both include and don't include dendrite segments.

    :param activations: 2D torch tensor with shape (batch_size, num_units) where entry
                        b, i gives the activation of unit i for example b
    :param targets: 1D torch tensor with shape (batch_size,) where entry b gives the
                    target label for example b
    """
    repr_overlaps = repr_overlap_stats(activations, targets)

    # Assume the following:
    # - target values are zero-based
    # - the largest target value in the batch is that amongst all data
    num_categories = 1 + targets.max().item()

    inter_class_overlaps = []
    intra_class_overlaps = []

    for t1 in range(num_categories):
        for t2 in range(1 + t1):

            if t1 == t2:
                intra_class_overlaps.extend(repr_overlaps[t1][t2])
            else:
                inter_class_overlaps.extend(repr_overlaps[t1][t2])

    return inter_class_overlaps, intra_class_overlaps


def repr_overlap_stats(activations, targets):
    """
    Returns `repr_overlaps`: a dict of dicts, where
    repr_overlaps[class_id_1][class_id_2] is list of representation overlap values
    between all pairs in class_id_1 and class_id_2; self-paired examples are excluded
    in the case where the two classes are the same.
    """
    with torch.no_grad():

        num_categories = 1 + targets.max().item()
        _, num_units = activations.size()

        repr_overlaps = {}

        for t1 in range(num_categories):

            repr_overlaps[t1] = {}

            for t2 in range(1 + t1):

                inds_1 = torch.nonzero((targets == t1).float(), as_tuple=True)
                activations_1 = activations[inds_1]
                activations_1 = (activations_1 != 0.0).float()

                inds_2 = torch.nonzero((targets == t2).float(), as_tuple=True)
                activations_2 = activations[inds_2]
                activations_2 = (activations_2 != 0.0).float()

                overlap_vals = torch.matmul(activations_1, activations_2.T)
                overlap_vals /= num_units

                # If `overlap_vals` gives intra-class representation overlaps, only
                # consider elements below the main diagonal to ignore duplicates and
                # identical pairs
                if t1 == t2:
                    num_examples, _ = overlap_vals.size()

                    overlap_vals = overlap_vals.tolist()
                    overlap_vals = [overlap_vals[x1][x2] for x1 in range(num_examples)
                                    for x2 in range(x1)]

                else:
                    overlap_vals = overlap_vals.flatten().tolist()

                repr_overlaps[t1][t2] = overlap_vals

        return repr_overlaps
