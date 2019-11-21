# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from .loggers import DSNNLogger
from .main import SparseModel

__all__ = [
    "DSNNHeb",
    "DSNNWeightedMag",
    "DSNNMixedHeb",
    "DSNNMixedHebInverse",
    "SET"
]


class DSNNHeb(SparseModel):
    """Parent class for DSNNHeb models. Not to be instantiated"""

    def setup(self):
        super().setup()

        # add specific defaults
        new_defaults = dict(
            pruning_active=True,
            pruning_early_stop=None,
            hebbian_prune_perc=None,
            weight_prune_perc=None,
            hebbian_grow=False,
            reset_coactivations=True,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # Set some attributes to be a list corresponding to self.dynamic_sparse_modules.
        for attr in ["hebbian_prune_perc", "weight_prune_perc"]:
            self._make_attr_iterable(attr, counterpart=self.sparse_modules)

        for idx, (hprune, wprune) in enumerate(
            zip(self.hebbian_prune_perc, self.weight_prune_perc)
        ):
            module = self.sparse_modules[idx]
            module.hebbian_prune = hprune
            module.weight_prune = wprune

        # initialize hebbian learning
        self._init_hebbian()
        self.prune_cycles_completed = 0

        self.logger = DSNNLogger(self, config=self.config)

    def _init_hebbian(self):
        for module in self.sparse_modules:
            if module.hebbian_prune:
                module.init_coactivation_tracking()
        if hasattr(self.network, "forward_with_coactivations"):
            self.network.forward = self.network.forward_with_coactivations

    def _is_dynamic(self, module):
        return True

    def _pre_epoch_setup(self):
        if self.reset_coactivations:
            for module in self.sparse_modules:
                if module.hebbian_prune:
                    module.reset_coactivations()

    def _post_epoch_updates(self, dataset=None):
        super()._post_epoch_updates(dataset)
        # zero out correlations (move to network)
        self._reinitialize_weights()
        # decide whether to stop pruning
        if self.pruning_early_stop:
            if self.current_epoch in self.lr_milestones:
                self.prune_cycles_completed += 1
                if self.prune_cycles_completed >= self.pruning_early_stop:
                    self.pruning_active = False

    def _reinitialize_weights(self):
        """Reinitialize weights - prune and grow"""
        if self.pruning_active:
            # keep track of added synapes
            for module in self.sparse_modules:
                if self._is_dynamic(module):
                    # prune
                    keep_mask = self.prune(module)
                    # grow
                    num_params = module.num_params
                    num_add = int(max(num_params - torch.sum(keep_mask).item(), 0))
                    add_mask = self.grow(module, num_add)
                    # join both
                    new_mask = keep_mask | add_mask
                    with torch.no_grad():
                        module.mask = new_mask.float()
                        module.apply_mask()

                    self.logger.save_masks(
                        module.pos, new_mask, keep_mask, add_mask, num_add
                    )
                    self.logger.save_surviving_synapses(module, keep_mask, add_mask)

    def _get_hebbian_mask(self, weight, corr, active_synapses, prune_perc):

        num_synapses = np.prod(weight.shape)
        total_active = torch.sum(active_synapses).item()

        corr_active = corr[active_synapses]
        # decide which weights to remove based on correlation
        kth = int(prune_perc * total_active)
        # if kth = 0, keep all the synapses
        if kth == 0:
            hebbian_mask = active_synapses
        # else if kth greater than shape, remove all synapses
        elif kth >= num_synapses:
            hebbian_mask = torch.zeros(weight.shape)
        # if no edge cases
        else:
            keep_threshold, _ = torch.kthvalue(corr_active, kth)
            # keep mask are ones above threshold and currently active
            hebbian_mask = (corr > keep_threshold) & active_synapses
        # move to device
        hebbian_mask.to(self.device)

        return hebbian_mask

    def _get_inverse_hebbian_mask(self, weight, corr, active_synapses, prune_perc):

        num_synapses = np.prod(weight.shape)
        total_active = torch.sum(active_synapses).item()

        corr_active = corr[active_synapses]
        # decide which weights to remove based on correlation
        kth = int((1 - prune_perc) * total_active)
        # if kth = 0, keep all the synapses
        if kth == 0:
            hebbian_mask = torch.zeros(weight.shape).bool()
        # else if kth greater than shape, remove all synapses
        elif kth >= num_synapses:
            hebbian_mask = active_synapses
        # if no edge cases
        else:
            keep_threshold, _ = torch.kthvalue(corr_active, kth)
            # keep mask are ones above threshold and currently active
            hebbian_mask = (corr <= keep_threshold) & active_synapses
        hebbian_mask = hebbian_mask.to(self.device)

        return hebbian_mask

    def _get_magnitude_mask(self, weight, active_synapses, prune_perc):

        # calculate the positive
        weight_pos = weight[weight > 0]
        pos_kth = int(prune_perc * len(weight_pos))
        # if no positive weight, threshold can be 0 (select none)
        if len(weight_pos) > 0:
            # if prune_perc=0, pos_kth=0, prune nothing
            if pos_kth == 0:
                pos_threshold = -1
            else:
                pos_threshold, _ = torch.kthvalue(weight_pos, pos_kth)
        else:
            pos_threshold = 0

        # calculate the negative
        weight_neg = weight[weight < 0]
        neg_kth = int((1 - prune_perc) * len(weight_neg))
        # if no negative weight, threshold -1 (select none)
        if len(weight_neg) > 0:
            # if prune_perc=1, neg_kth=0, prune all
            if neg_kth == 0:
                neg_threshold = torch.min(weight_neg).item() - 1
            else:
                neg_threshold, _ = torch.kthvalue(weight_neg, neg_kth)
        else:
            neg_threshold = -1

        # consolidate
        partial_weight_mask = (weight > pos_threshold) | (weight <= neg_threshold)
        weight_mask = partial_weight_mask & active_synapses
        weight_mask = weight_mask.to(self.device)

        return weight_mask

    def _get_random_add_mask_prob(self, nonactive_synapses, num_add):
        """
        Deprecated method of add random mask.
        Faster, but adds stochasticity to number of added params - tricky to test
        """
        total_nonactive = torch.sum(nonactive_synapses).item()
        p_add = num_add / max(total_nonactive, num_add)
        random_sample = torch.rand(nonactive_synapses.shape).to(self.device) < p_add
        add_mask = random_sample & nonactive_synapses

        return add_mask

    def _get_random_add_mask(self, nonactive_synapses, num_add):
        """
        Random mask that ensures the exact number of params is added
        For computationally faster method, see _get_random_add_mask_prob
        """
        nonzero = torch.nonzero(nonactive_synapses, as_tuple=False)
        num_add = min(int(num_add), len(nonzero))  # can't select more than available
        sampled_idxs = np.random.choice(range(len(nonzero)), num_add, replace=False)
        selected = nonzero[sampled_idxs]

        add_mask = torch.zeros(nonactive_synapses.shape, dtype=torch.bool)
        if len(selected.shape) > 1:
            add_mask[list(zip(*selected))] = True
        else:
            add_mask[selected] = True

        return add_mask.to(self.device)

    def _get_hebbian_add_mask(self, corr, nonactive_synapses, num_add):

        # get threshold
        total_nonactive = torch.sum(nonactive_synapses).item()
        kth = int(total_nonactive - num_add)  # should not be non-int
        corr_nonactive = corr[nonactive_synapses]
        add_threshold, _ = torch.kthvalue(corr_nonactive, kth)
        # calculate mask, only for currently nonactive
        add_mask = (corr > add_threshold) & nonactive_synapses

        return add_mask

    def _get_inverse_add_mask(self, corr, nonactive_synapses, num_add):

        # get threshold
        kth = int(num_add)  # should not be non-int
        if kth > 0:
            corr_nonactive = corr[nonactive_synapses]
            add_threshold, _ = torch.kthvalue(corr_nonactive, kth)
            # calculate mask, only for currently nonactive
            add_mask = (corr <= add_threshold) & nonactive_synapses
        # if there is nothing to add, return zeros
        else:
            add_mask = torch.zeros(nonactive_synapses.shape).bool()

        return add_mask

    def prune(self, module):
        pass


class DSNNWeightedMag(DSNNHeb):
    """Weight weights using correlation"""

    def _is_dynamic(self, module):
        return module.hebbian_prune is not None

    def prune(self, module):
        """Prune by magnitude"""
        with torch.no_grad():
            # unpack module
            weight = module.m.weight.clone().detach()
            corr = module.get_coactivations()
            hebbian_prune_perc = module.hebbian_prune
            # init shared variables
            active_synapses = weight != 0

            if hebbian_prune_perc is not None:
                # multiply correlation by weight, and then apply regular weight pruning
                weight *= corr
                keep_mask = self._get_magnitude_mask(
                    weight, active_synapses, hebbian_prune_perc
                )
            else:
                keep_mask = active_synapses.to(self.device)

        return keep_mask

    def grow(self, module, num_add):
        """Add randomly"""
        with torch.no_grad():
            nonactive_synapses = module.m.weight == 0
            add_mask = self._get_random_add_mask(nonactive_synapses, num_add)
            add_mask = add_mask.to(self.device)

        return add_mask


class DSNNMixedHeb(DSNNHeb):
    """Improved results compared to DSNNHeb"""

    def _is_dynamic(self, module):
        return module.hebbian_prune is not None or module.weight_prune is not None

    def prune(self, module):
        """Allows pruning by magnitude and hebbian"""
        with torch.no_grad():
            # unpack module
            weight = module.m.weight.clone().detach()
            hebbian_prune_perc = module.hebbian_prune
            weight_prune_perc = module.weight_prune
            active_synapses = weight != 0

            # hebbian mask
            if hebbian_prune_perc is not None:
                corr = module.get_coactivations()
                hebbian_mask = self._get_hebbian_mask(
                    weight, corr, active_synapses, hebbian_prune_perc
                )
            else:
                hebbian_mask = None

            # weight mask
            if weight_prune_perc is not None:
                magnitude_mask = self._get_magnitude_mask(
                    weight, active_synapses, weight_prune_perc
                )
            else:
                magnitude_mask = None

            # join both masks
            if hebbian_prune_perc and weight_prune_perc:
                keep_mask = hebbian_mask | magnitude_mask
            elif hebbian_prune_perc:
                keep_mask = hebbian_mask
            elif weight_prune_perc:
                keep_mask = magnitude_mask
            # if no pruning, just perpetuate same synapses
            else:
                keep_mask = active_synapses.to(self.device)

        return keep_mask

    def grow(self, module, num_add):
        """Add randomly"""
        with torch.no_grad():
            nonactive_synapses = module.m.weight == 0
            if self.hebbian_grow:
                corr = module.get_coactivations()
                add_mask = self._get_hebbian_add_mask(corr, nonactive_synapses, num_add)
            else:
                add_mask = self._get_random_add_mask(nonactive_synapses, num_add)
            add_mask = add_mask.to(self.device)

        return add_mask


class SET(DSNNHeb):
    """Improved results compared to DSNNHeb"""

    def _is_dynamic(self, module):
        return module.weight_prune is not None

    def prune(self, module):
        """Allows pruning by magnitude and hebbian"""
        with torch.no_grad():
            # unpack module
            weight = module.m.weight.clone().detach()
            weight_prune_perc = module.weight_prune
            active_synapses = weight != 0

            # weight mask
            if weight_prune_perc is not None:
                keep_mask = self._get_magnitude_mask(
                    weight, active_synapses, weight_prune_perc
                )
            else:
                keep_mask = active_synapses

        return keep_mask.to(self.device)

    def grow(self, module, num_add):
        """Add randomly"""
        with torch.no_grad():
            nonactive_synapses = module.m.weight == 0
            add_mask = self._get_random_add_mask(nonactive_synapses, num_add)
            add_mask = add_mask.to(self.device)

        return add_mask


class DSNNMixedHebInverse(DSNNMixedHeb):
    """Test the extreme alternative hypothesis"""

    def prune(self, module):
        """Allows pruning by magnitude and hebbian"""
        with torch.no_grad():
            # unpack module
            weight = module.m.weight.clone().detach()
            hebbian_prune_perc = module.hebbian_prune
            weight_prune_perc = module.weight_prune
            # init shared variables
            active_synapses = weight != 0

            # hebbian mask
            if hebbian_prune_perc is not None:
                corr = module.get_coactivations()
                hebbian_mask = self._get_inverse_hebbian_mask(
                    weight, corr, active_synapses, hebbian_prune_perc
                )
            else:
                hebbian_mask = None

            # weight mask
            if weight_prune_perc is not None:
                magnitude_mask = self._get_magnitude_mask(
                    weight, active_synapses, weight_prune_perc
                )
            else:
                magnitude_mask = None

            # join both masks
            if hebbian_prune_perc and weight_prune_perc:
                keep_mask = hebbian_mask | magnitude_mask
            elif hebbian_prune_perc:
                keep_mask = hebbian_mask
            elif weight_prune_perc:
                keep_mask = magnitude_mask
            # if no pruning, just perpetuate same synapses
            else:
                keep_mask = active_synapses.to(self.device)

        return keep_mask

    def grow(self, module, num_add):
        """Add randomly"""
        with torch.no_grad():
            nonactive_synapses = module.m.weight == 0
            if self.hebbian_grow:
                corr = module.get_coactivations()
                add_mask = self._get_inverse_hebbian_add_mask(
                    corr, nonactive_synapses, num_add
                )
            else:
                add_mask = self._get_random_add_mask(nonactive_synapses, num_add)
            add_mask = add_mask.to(self.device)

        return add_mask
