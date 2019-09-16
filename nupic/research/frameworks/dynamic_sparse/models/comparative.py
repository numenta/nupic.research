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

import torch

from nupic.research.frameworks.dynamic_sparse.networks import DynamicSparseBase

from .main import SparseModel


class SET(SparseModel):
    """
    Implementation of SET with a more efficient approach of adding new
    weights (vectorized) The overhead in computation is 10x smaller compared to
    the original version.
    """

    def setup(self):
        super().setup()
        # initialize data structure keep track of added synapses
        self.added_synapses = [None for m in self.masks]

        # add early stopping to SET
        self.pruning_active = True
        self.prune_cycles_completed = 0

    def _post_epoch_updates(self, dataset):
        super()._post_epoch_updates(dataset)
        self._reinitialize_weights()
        # decide whether to stop pruning
        if self.pruning_early_stop:
            if self.current_epoch in self.lr_milestones:
                self.prune_cycles_completed += 1
                if self.prune_cycles_completed >= self.pruning_early_stop:
                    self.pruning_active = False
        # temporarily logging for debug purposes
        self.log["pruning_early_stop"] = int(self.pruning_early_stop)

    def _reinitialize_weights(self):
        """Reinitialize weights."""
        if self.pruning_active:
            for idx, m in enumerate(self.sparse_modules):
                layer_weights = m.weight.clone().detach()
                new_mask, prune_mask, new_synapses = self.prune(
                    layer_weights, self.num_params[idx]
                )
                with torch.no_grad():
                    self.masks[idx] = new_mask.float()
                    m.weight.data *= prune_mask.float()

                    # keep track of added synapes
                    if self.debug_sparse:
                        self.log["added_synapses_l" + str(idx)] = torch.sum(
                            new_synapses
                        ).item()
                        if self.added_synapses[idx] is not None:
                            total_added = torch.sum(self.added_synapses[idx]).item()
                            surviving = torch.sum(
                                self.added_synapses[idx] & prune_mask
                            ).item()
                            if total_added:
                                self.log["surviving_synapses_l" + str(idx)] = (
                                    surviving / total_added
                                )
                    self.added_synapses[idx] = new_synapses

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            for idx, m in enumerate(self.masks):
                self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()

    def prune(self, weight, num_params, zeta=0.3):
        """
        Calculate new weight based on SET approach weight vectorized version
        aimed at keeping the mask with the similar level of sparsity.
        """
        with torch.no_grad():

            # NOTE: another approach is counting how many numbers to prune
            # calculate thresholds and decay weak connections
            weight_pos = weight[weight > 0]
            pos_threshold, _ = torch.kthvalue(weight_pos, int(zeta * len(weight_pos)))
            weight_neg = weight[weight < 0]
            neg_threshold, _ = torch.kthvalue(
                weight_neg, int((1 - zeta) * len(weight_neg))
            )
            prune_mask = ((weight >= pos_threshold) | (weight <= neg_threshold)).to(
                self.device
            )

            # change mask to add new weight
            num_add = num_params - torch.sum(prune_mask).item()
            current_sparsity = torch.sum(weight == 0).item()
            p_add = num_add / max(current_sparsity, num_add)  # avoid div by zero
            probs = torch.rand(weight.shape).to(self.device) < p_add
            new_synapses = probs & (weight == 0)
            new_mask = prune_mask | new_synapses

        # track added connections
        return new_mask, prune_mask, new_synapses


class DynamicRep(SparseModel):
    """REMINDER: need to remove downsampling layers - figure out how to do"""

    def setup(self):
        super().setup()
        # define sparsity - more intuitive than on_perc
        self.h_tolerance = 0.05
        self.zeta = 0.2
        # count params
        self._initialize_prune_threshold()
        self._count_params()

        # debugging start
        toprune_params = int(self.available_params * (self.zeta))
        toprune_baseline = (
            int(toprune_params * (1 - self.h_tolerance)),
            int(toprune_params * (1 + self.h_tolerance)),
        )
        print(toprune_params)
        print(toprune_baseline)

        # initialize data structure keep track of added synapses
        self.added_synapses = [None for m in self.masks]

    def _count_params(self):
        """
        Count parameters of the network (sparse modules alone)
        No need to keep track of full parameters, just the available ones
        """
        self.available_params = 0
        for m in list(self.network.modules()):
            if isinstance(m, DynamicSparseBase):
                self.available_params += torch.sum(m.weight != 0).item()

    def _initialize_prune_threshold(self):
        """Initialize prune threshold h"""
        weighted_mean = 0
        total_params_count = 0
        # initialize h and total_params
        with torch.no_grad():
            for m in list(self.network.modules()):
                if isinstance(m, DynamicSparseBase):
                    # count how many weights are not equal to 0
                    count_p = torch.sum(m.weight != 0).item()
                    # get topk for that level, and weight by num of values
                    non_zero = torch.abs(m.weight[m.weight != 0]).view(-1)
                    val, _ = torch.kthvalue(non_zero, int(len(non_zero) * self.on_perc))
                    weighted_mean += count_p * val.item()
                    total_params_count += count_p

        # get initial value for h based on enforced sparsity
        self.h = weighted_mean / total_params_count
        print(self.h)

    def _run_one_pass(self, loader, train, noise=False):
        super()._run_one_pass(loader, train, noise)
        if train:
            self.reinitialize_weights()

    def reinitialize_weights(self):
        """Steps
        1- calculate how many weights should go to the layer
        2- call prune on each layer
        3- update H
        """
        with torch.no_grad():
            total_available = self.available_params
            # count to prune based on a fixed percentage of the surviving weights
            toprune_count = int(self.zeta * total_available)
            self.pruned_count = 0
            self.grown_count = 0
            for idx, m in enumerate(self.sparse_modules):
                # calculate number of weights to add
                available = torch.sum(m.weight != 0).item()
                num_add = int(available / total_available * toprune_count)
                # prune weights
                new_mask, keep_mask, grow_mask = self.prune_and_grow(
                    m.weight.clone().detach(), num_add
                )
                self.masks[idx] = new_mask.float()
                m.weight.data *= self.masks[idx].float()

                # DEBUGGING STUFF. TODO: move code to some other place

                # count how many synapses from last round have survived
                if self.added_synapses[idx] is not None:
                    total_added = torch.sum(self.added_synapses[idx]).item()
                    surviving = torch.sum(self.added_synapses[idx] & keep_mask).item()
                    if total_added:
                        survival_ratio = surviving / total_added
                        # log if in debug sparse mode
                        if self.debug_sparse:
                            self.log["surviving_synapses_l" + str(idx)] = survival_ratio

                # keep track of new synapses to count surviving on next round
                self.added_synapses[idx] = grow_mask

        # track main parameters
        self.log["dyre_total_available"] = total_available
        self.log["dyre_delta_available"] = self.available_params - total_available
        self.log["dyre_to_prune"] = toprune_count
        self.log["dyre_pruned_count"] = self.pruned_count
        self.log["dyre_grown_count"] = self.grown_count
        self.log["dyre_h_threshold"] = self.h
        # update H according to the simple rule
        # if actual pruned less than threshold, reduce tolerance
        if self.pruned_count < int(toprune_count * (1 - self.h_tolerance)):
            self.h *= 2
            self.log["dyre_h_delta"] = 1
        # if greater, increase tolerance
        elif self.pruned_count > int(toprune_count * (1 + self.h_tolerance)):
            self.h /= 2
            self.log["dyre_h_delta"] = -1

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            for idx, m in enumerate(self.masks):
                self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()

    def prune_and_grow(self, weight, num_add):
        """Steps
        1- Sample positions to grow new weights (sample indexes)
        2- Update parameter count
        3- Prune parameters based on H - update prune mask
        4- Grow randomly sampled parameters - update add mask
        5 - return both masks
        """
        with torch.no_grad():

            # GROW
            # identify non-zero entries
            nonzero_idxs = torch.nonzero(weight)
            num_available_params = len(nonzero_idxs)
            # randomly sample
            random_sample = torch.randperm(len(nonzero_idxs))[:num_add]
            togrow_idxs = nonzero_idxs[random_sample]
            # update the mask with added weights
            grow_mask = torch.zeros(weight.shape, dtype=torch.bool).to(self.device)
            grow_mask[tuple(togrow_idxs.T)] = 1
            num_grow = len(togrow_idxs)

            # PRUNE

            keep_mask = (torch.abs(weight) > self.h).to(self.device)
            num_prune = num_available_params - torch.sum(keep_mask).item()

            # combine both
            new_mask = keep_mask | grow_mask

            # update parameter count
            self.pruned_count += num_prune
            self.grown_count += num_grow
            self.available_params += num_grow
            self.available_params -= num_prune

        # track added connections
        return new_mask, keep_mask, grow_mask
