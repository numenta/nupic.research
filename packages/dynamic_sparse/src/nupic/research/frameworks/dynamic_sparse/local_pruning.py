# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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

from nupic.research.frameworks.pytorch.mask_utils import get_topk_submask

__all__ = [
    "local_prune_by_abs_weight",
    "local_add_by_abs_grad",
]


def local_prune_by_abs_weight(sparse_modules, prune_fraction):
    """
    Prune `prune_fraction` of the weigths of each module in `sparse_modules`.

    Modules are pruned by ranking and selecting the top absolute weights of each
    module and adjusting the sparse module `zero_mask`s. Be sure to call
    `rezero_weights` following this function to zero out the pruned weights.

    :param sparse_modules: list of modules of type SparseWeightsBase
    :param prune_fraction: fraction of weights to prune; between 0 and 1

    :return: list containing the number of weights removed for each `sparse_module`
    """
    assert 0 <= prune_fraction <= 1

    total_removed = []
    for module in sparse_modules:
        params = module.weight.detach().flatten()
        off_mask = module.zero_mask.detach().flatten().bool()

        # Compute new top K value
        on_mask = ~off_mask
        total_on = on_mask.sum()
        num_remove = (total_on * prune_fraction).round()
        k = int(total_on - num_remove)

        # Prune by only keeping the top weights ranked by their absolute values.
        topk_mask = get_topk_submask(
            k=k, values=params.abs(), mask=on_mask, largest=True
        )

        # Update module with new mask
        module.zero_mask[:] = (~topk_mask).view_as(module.weight)
        total_removed.append(int(num_remove))

    return total_removed


def local_add_by_abs_grad(sparse_modules, num_add):
    """
    Adds `num_add` weights distributed across the modules by ranking and
    selecting the top absolute gradients for each `sparse_module`. Weights are
    added by adjusting the sparse module `zero_masks`. Be sure to call
    `rezero_weights` following this function to initialize the new weights to
    zero.

    :param sparse_modules: list of modules of type SparseWeightsBase
    :param num_add: list with number of weights to add per sparse_module,
                    usually the output of :meth:`local_prune_by_abs_weight`
    """
    for i, module in enumerate(sparse_modules):
        grads = module.weight.grad.flatten()
        mask = module.zero_mask.detach().flatten().bool()

        # Find a mask of the top grads for weights that are currently off.
        topk_mask = get_topk_submask(
            k=num_add[i], values=grads.abs(), mask=mask, largest=True
        )
        module.zero_mask[topk_mask.view_as(module.weight)] = 0.
