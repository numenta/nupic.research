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

import numpy as np
import torch


def percent_active_dendrites(
    dendrite_weights,
    context_vectors,
    selection_criterion
):
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
