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

from torch.nn.utils import parameters_to_vector

from nupic.research.frameworks.pytorch.mask_utils import get_topk_submask

__all__ = [
    "global_prune_by_abs_weight",
    "global_add_by_abs_grad",
]


def global_prune_by_abs_weight(sparse_modules, prune_fraction=None, num_remove=None):
    """
    Globally prune 'num_remove' weights from a list of sparse modules by ranking and
    selecting the top absolute weights. If prune_fraction is given, then num_removed is
    calculated. Modules are pruned adjusting their `zero_masks`; switching off the
    weight occurs by changing 0 to 1. Be sure to call `rezero_weights` following this
    function to zero out the pruned weights.

    :param sparse_modules: list of modules of type SparseWeightsBase
    :param prune_fraction: fraction of weights to prune; between 0 and 1; can't be
                           specified if num_remove is not None
    :param num_remove: how many parameters to remove; can't be specified if
                       prune_fraction is not None
    """

    # Only one of these arguments may be given.
    assert not (prune_fraction is not None and num_remove is not None)

    # Flatten parameters to compare them all at once for global pruning.
    flattened_params = parameters_to_vector([m.weight for m in sparse_modules])
    flattened_off_mask = parameters_to_vector([m.zero_mask for m in sparse_modules])
    flattened_on_mask = ~flattened_off_mask.bool()

    # Calculate the number of parameters to keep.
    total_on = flattened_on_mask.sum().item()
    if prune_fraction is not None:
        assert 0 <= prune_fraction <= 1
        num_remove = int(round(total_on * prune_fraction))
    num_keep = total_on - num_remove

    # Prune by only keeping the top weights ranked by their absolute values.
    topk_mask = get_topk_submask(
        k=num_keep,
        values=flattened_params.abs(),
        mask=flattened_on_mask,
        largest=True
    )

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module in sparse_modules:

        num_params = module.weight.numel()
        keep_mask = topk_mask[pointer: pointer + num_params].view_as(module.weight)
        module.zero_mask[:] = ~keep_mask
        pointer += num_params

    return num_remove


def global_add_by_abs_grad(sparse_modules, num_add):
    """
    Adds weights globally (among all given sparse modules) by ranking and selecting the
    top absolute gradients. Weights are added by adjusting the `zero_masks` of their
    modules; switching on the weight occurs by changing 1 to 0. Be sure to call
    `rezero_weights` following this function to initialize the new weights to zero.

    :param sparse_modules: list of modules of type SparseWeightsBase
    :param num_add: number of weights to add
    """

    flattened_grads = parameters_to_vector([m.weight.grad for m in sparse_modules])
    flattened_mask = parameters_to_vector([m.zero_mask for m in sparse_modules]).bool()

    # Find a mask of the top grads for weights that are currently off.
    topk_mask = get_topk_submask(
        k=num_add,
        values=flattened_grads.abs(),
        mask=flattened_mask,
        largest=True
    )

    # Pointer for slicing the mask to match the shape of each parameter.
    pointer = 0
    for module in sparse_modules:

        num_params = module.weight.numel()
        add_mask = topk_mask[pointer: pointer + num_params].view_as(module.weight)
        module.zero_mask[add_mask] = 0
        pointer += num_params
