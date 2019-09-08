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

from collections import deque
from collections.abc import Iterable

import numpy as np
import torch

from nupic.research.frameworks.dynamic_sparse.networks.layers import (
    DSConv2d,
    SparseConv2d,
    calc_sparsity,
)

from .main import BaseModel, SparseModel


class DSNNHeb(SparseModel):
    """Improved results compared to regular SET"""

    def setup(self):
        super().setup()
        self.added_synapses = [None for m in self.masks]
        self.last_gradients = [None for m in self.masks]

        # add specific defaults
        new_defaults = dict(
            pruning_active=True,
            pruning_early_stop=None,
            hebbian_prune_perc=None,
            weight_prune_perc=None,
            hebbian_grow=True,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # initialize hebbian learning
        self.network.hebbian_learning = True
        self.prune_cycles_completed = 0

    def _post_epoch_updates(self, dataset=None):
        super()._post_epoch_updates(dataset)
        # zero out correlations (move to network)
        self._reinitialize_weights()
        self.network.correlations = []
        # decide whether to stop pruning
        if self.pruning_early_stop:
            if self.current_epoch in self.lr_milestones:
                self.prune_cycles_completed += 1
                if self.prune_cycles_completed >= self.pruning_early_stop:
                    self.pruning_active = False
        # temporarily logging for debug purposes
        self.log["pruning_early_stop"] = int(self.pruning_early_stop)

    def _reinitialize_weights(self):
        """Reinitialize weights - prune and grow"""
        if self.pruning_active:
            # keep track of added synapes
            survival_ratios = []

            for idx, (m, corr) in enumerate(
                zip(self.sparse_modules, self.network.correlations)
            ):
                new_mask, keep_mask, new_synapses = self.prune(
                    m.weight.clone().detach(), self.num_params[idx], corr, idx=idx
                )
                with torch.no_grad():
                    self.masks[idx] = new_mask.float()
                    m.weight.data *= self.masks[idx]

                    # count how many synapses from last round have survived
                    if self.added_synapses[idx] is not None:
                        total_added = torch.sum(self.added_synapses[idx]).item()
                        surviving = torch.sum(
                            self.added_synapses[idx] & keep_mask
                        ).item()
                        if total_added:
                            survival_ratio = surviving / total_added
                            survival_ratios.append(survival_ratio)

                    # keep track of new synapses to count surviving on next round
                    self.added_synapses[idx] = new_synapses

            # logging 
            if self.debug_sparse:
                for idx, (m, sr) in enumerate(zip(self.masks, survival_ratios)):
                    self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()
                    self.log["surviving_synapses_l" + str(idx)] = sr
                self.log["surviving_synapses_avg"] = np.mean(survival_ratios)

    def prune(self, weight, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by correlation and magnitude
        """
        with torch.no_grad():

            # calculate weight mask
            zeta = self.weight_prune_perc
            weight_pos = weight[weight > 0]
            pos_threshold, _ = torch.kthvalue(
                weight_pos, max(int(zeta * len(weight_pos)), 1)
            )
            weight_neg = weight[weight < 0]
            neg_threshold, _ = torch.kthvalue(
                weight_neg, max(int((1 - zeta) * len(weight_neg)), 1)
            )
            weight_keep_mask = (weight >= pos_threshold) | (weight <= neg_threshold)
            weight_keep_mask.to(self.device)

            # no gradient mask, just a keep mask
            keep_mask = weight_keep_mask

            # calculate number of parameters to add
            num_add = num_params - torch.sum(keep_mask).item()
            # transpose to fit the weights
            corr = corr.t()
            # remove the ones which will already be kept
            corr *= (keep_mask == 0).float()
            # get kth value, based on how many weights to add, and calculate mask
            kth = int(np.prod(corr.shape) - num_add)
            # contiguous()
            corr_threshold, _ = torch.kthvalue(corr.contiguous().view(-1), kth)
            add_mask = (corr > corr_threshold).to(self.device)

            new_mask = keep_mask | add_mask

            # logging
            if self.debug_sparse:
                self.log["weight_keep_mask_l" + str(idx)] = torch.sum(
                    weight_keep_mask
                ).item()
                self.log["missing_weights_l" + str(idx)] = num_add
                self.log["added_synapses_l" + str(idx)] = torch.sum(add_mask).item()

        # track added connections
        return new_mask, keep_mask, add_mask


class DSNNWeightedMag(DSNNHeb):
    """Weight weights using correlation"""

    def prune(self, weight, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by magnitude
        """
        with torch.no_grad():
            # init shared variables
            num_synapses = np.prod(weight.shape)
            active_synapses = (weight != 0)
            nonactive_synapses = (weight == 0)
            total_active = torch.sum(active_synapses).item()
            total_nonactive = torch.sum(nonactive_synapses).item()
            zeta = self.weight_prune_perc
            # transpose correlation to the weight matrix
            corr = corr.t()
            # multiply correlation by weight, and then apply regular weight pruning
            weight *= corr

            # ----------- WEIGHT PRUNING ----------------
                            
            if zeta is not None:
                # calculate the positive
                weight_pos = weight[weight > 0]
                pos_kth = int(zeta * len(weight_pos))
                # if zeta=0, pos_kth=0, prune nothing
                if pos_kth == 0:
                    pos_threshold = -1
                else:
                    pos_threshold, _ = torch.kthvalue(weight_pos, pos_kth)
                
                # calculate the negative
                weight_neg = weight[weight < 0]
                neg_kth = int((1-zeta) * len(weight_neg))
                # if zeta=1, neg_kth=0, prune all
                if neg_kth == 0:
                    neg_threshold = torch.min(weight_neg).item() - 1
                else:
                    neg_threshold, _ = torch.kthvalue(weight_neg, neg_kth)
                    
                partial_weight_mask = (weight > pos_threshold) | (weight <= neg_threshold)
                weight_mask = partial_weight_mask & active_synapses
                # move to device
                weight_mask.to(self.device)

            # ----------- COMBINE HEBBIAN AND WEIGHT ----------------            
            if zeta:
                keep_mask = weight_mask
            else:
                keep_mask = active_synapses.to(self.device)

            # ----------- GROWTH ----------------      

            num_add = max(num_params - torch.sum(keep_mask).item(), 0)  
            # probability of adding is 1 or lower
            p_add = num_add / max(total_nonactive, num_add)
            random_sample = torch.rand(weight.shape).to(self.device) < p_add
            add_mask = random_sample & nonactive_synapses

            # calculate the new mask
            new_mask = keep_mask | add_mask

            # logging
            if self.debug_sparse:
                self.log["keep_mask_l" + str(idx)] = (
                    torch.sum(keep_mask).item() / num_synapses
                )
                self.log["add_mask_l" + str(idx)] = (
                    torch.sum(add_mask).item() / num_synapses
                )
                self.log["missing_weights_l" + str(idx)] = num_add / num_synapses
                # conditional logs
                if zeta is not None:
                    self.log["weight_mask_l" + str(idx)] = (
                        torch.sum(weight_mask).item() / num_synapses
                    )

        # track added connections
        return new_mask, keep_mask, add_mask


class DSNNMixedHeb(DSNNHeb):
    """Improved results compared to DSNNHeb"""


    def prune_inverse(self, weight, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by magnitude
        """
        with torch.no_grad():
            # init shared variables
            num_synapses = np.prod(weight.shape)
            active_synapses = (weight != 0)
            nonactive_synapses = (weight == 0)
            total_active = torch.sum(active_synapses).item()
            total_nonactive = torch.sum(nonactive_synapses).item()
            tau = self.hebbian_prune_perc
            zeta = self.weight_prune_perc
            # transpose correlation to the weight matrix
            corr = corr.t()

            # ----------- HEBBIAN PRUNING ----------------
            
            if tau:
                corr_active = corr[active_synapses]
                # decide which weights to remove based on correlation
                kth = int((1-tau) * total_active)
                # if kth = 0, keep all the synapses
                if kth == 0:
                    hebbian_keep_mask = torch.zeros(weight.shape).bool()
                # else if kth greater than shape, remove all synapses
                elif kth >= num_synapses:
                    hebbian_keep_mask = active_synapses
                # if no edge cases
                else:
                    keep_threshold, _ = torch.kthvalue(corr_active, kth)
                    # keep mask are ones above threshold and currently active
                    hebbian_keep_mask = (corr <= keep_threshold) & active_synapses
                hebbian_keep_mask = hebbian_keep_mask.to(self.device)

            # ----------- WEIGHT PRUNING ----------------
                            
            if zeta:
                # calculate the positive
                weight_pos = weight[weight > 0]
                pos_kth = int(zeta * len(weight_pos))
                # if no positive weight, threshold can be 0 (select none)
                if len(weight_pos) > 0:
                    # if zeta=0, pos_kth=0, prune nothing
                    if pos_kth == 0:
                        pos_threshold = -1
                    else:
                        pos_threshold, _ = torch.kthvalue(weight_pos, pos_kth)
                else:
                    pos_threshold = 0
                
                # calculate the negative
                weight_neg = weight[weight < 0]
                neg_kth = int((1-zeta) * len(weight_neg))
                # if no negative weight, threshold -1 (select none)
                if len(weight_neg) > 0:
                    # if zeta=1, neg_kth=0, prune all
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

            # ----------- COMBINE HEBBIAN AND WEIGHT ----------------            
                
            # join both masks
            if tau and zeta:
                keep_mask = hebbian_keep_mask | weight_mask
            elif tau:
                keep_mask = hebbian_keep_mask
            elif zeta:
                keep_mask = weight_mask
            else:
                keep_mask = active_synapses.to(self.device)

            # ----------- GROWTH ----------------      

            num_add = max(num_params - torch.sum(keep_mask).item(), 0)  
            # added option to have hebbian grow or not
            if self.hebbian_grow:
                # get threshold
                kth = int(num_add) # should not be non-int
                if kth > 0:
                    corr_nonactive = corr[nonactive_synapses]
                    add_threshold, _ = torch.kthvalue(corr_nonactive, kth)
                    # calculate mask, only for currently nonactive
                    add_mask = (corr <= add_threshold) & nonactive_synapses
                # if there is nothing to add, return zeros
                else:
                    add_mask = torch.zeros(weight.shape).bool()
            else:
                # probability of adding is 1 or lower
                p_add = num_add / max(total_nonactive, num_add)
                random_sample = torch.rand(weight.shape).to(self.device) < p_add
                add_mask = random_sample & nonactive_synapses
            add_mask = add_mask.to(self.device)

            # calculate the new mask
            new_mask = keep_mask | add_mask

            # logging
            if self.debug_sparse:
                self.log["keep_mask_l" + str(idx)] = (
                    torch.sum(keep_mask).item() / num_synapses
                )
                self.log["add_mask_l" + str(idx)] = (
                    torch.sum(add_mask).item() / num_synapses
                )
                self.log["missing_weights_l" + str(idx)] = num_add / num_synapses
                # conditional logs
                if tau is not None:
                    self.log["hebbian_keep_mask_l" + str(idx)] = (
                        torch.sum(hebbian_keep_mask).item() / num_synapses
                    )
                if zeta is not None:
                    self.log["weight_mask_l" + str(idx)] = (
                        torch.sum(weight_mask).item() / num_synapses
                    )

        # track added connections
        return new_mask, keep_mask, add_mask

    def prune(self, weight, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by magnitude
        """
        with torch.no_grad():
            # init shared variables
            num_synapses = np.prod(weight.shape)
            active_synapses = (weight != 0)
            nonactive_synapses = (weight == 0)
            total_active = torch.sum(active_synapses).item()
            total_nonactive = torch.sum(nonactive_synapses).item()
            tau = self.hebbian_prune_perc
            zeta = self.weight_prune_perc
            # transpose correlation to the weight matrix
            corr = corr.t()

            # ----------- HEBBIAN PRUNING ----------------
            
            if tau is not None:
                corr_active = corr[active_synapses]
                # decide which weights to remove based on correlation
                kth = int(tau * total_active)
                # if kth = 0, keep all the synapses
                if kth == 0:
                    hebbian_keep_mask = active_synapses
                # else if kth greater than shape, remove all synapses
                elif kth >= num_synapses:
                    hebbian_keep_mask = torch.zeros(weight.shape)
                # if no edge cases
                else:
                    keep_threshold, _ = torch.kthvalue(corr_active, kth)
                    # keep mask are ones above threshold and currently active
                    hebbian_keep_mask = (corr > keep_threshold) & active_synapses
                # move to device
                hebbian_keep_mask.to(self.device)

            # ----------- WEIGHT PRUNING ----------------
                            
            if zeta is not None:
                # calculate the positive
                weight_pos = weight[weight > 0]
                pos_kth = int(zeta * len(weight_pos))
                # if zeta=0, pos_kth=0, prune nothing
                if pos_kth == 0:
                    pos_threshold = -1
                else:
                    pos_threshold, _ = torch.kthvalue(weight_pos, pos_kth)
                
                # calculate the negative
                weight_neg = weight[weight < 0]
                neg_kth = int((1-zeta) * len(weight_neg))
                # if zeta=1, neg_kth=0, prune all
                if neg_kth == 0:
                    neg_threshold = torch.min(weight_neg).item() - 1
                else:
                    neg_threshold, _ = torch.kthvalue(weight_neg, neg_kth)
                    
                partial_weight_mask = (weight > pos_threshold) | (weight <= neg_threshold)
                weight_mask = partial_weight_mask & active_synapses
                # move to device
                weight_mask.to(self.device)

            # ----------- COMBINE HEBBIAN AND WEIGHT ----------------            
                
            # join both masks
            if tau and zeta:
                keep_mask = hebbian_keep_mask | weight_mask
            elif tau:
                keep_mask = hebbian_keep_mask
            elif zeta:
                keep_mask = weight_mask
            else:
                keep_mask = active_synapses.to(self.device)

            # ----------- GROWTH ----------------      

            num_add = max(num_params - torch.sum(keep_mask).item(), 0)  
            # added option to have hebbian grow or not
            if self.hebbian_grow:
                # get threshold
                kth = int(total_nonactive - num_add) # should not be non-int
                corr_nonactive = corr[nonactive_synapses]
                add_threshold, _ = torch.kthvalue(corr_nonactive, kth)
                # calculate mask, only for currently nonactive
                add_mask = (corr > add_threshold) & nonactive_synapses
            else:
                # probability of adding is 1 or lower
                p_add = num_add / max(total_nonactive, num_add)
                random_sample = torch.rand(weight.shape).to(self.device) < p_add
                add_mask = random_sample & nonactive_synapses

            # calculate the new mask
            new_mask = keep_mask | add_mask

            # logging
            if self.debug_sparse:
                self.log["keep_mask_l" + str(idx)] = (
                    torch.sum(keep_mask).item() / num_synapses
                )
                self.log["add_mask_l" + str(idx)] = (
                    torch.sum(add_mask).item() / num_synapses
                )
                self.log["missing_weights_l" + str(idx)] = num_add / num_synapses
                # conditional logs
                if tau is not None:
                    self.log["hebbian_keep_mask_l" + str(idx)] = (
                        torch.sum(hebbian_keep_mask).item() / num_synapses
                    )
                if zeta is not None:
                    self.log["weight_mask_l" + str(idx)] = (
                        torch.sum(weight_mask).item() / num_synapses
                    )

        # track added connections
        return new_mask, keep_mask, add_mask


class DSNNConvHeb(DSNNMixedHeb):
    """
    Similar to other sparse models, but the focus here is on convolutional layers as
    opposed to dense layers.
    """

    log_attrs = [
        "pruning_iterations",
        "kept_frac",
        "prune_mask_sparsity",
        "keep_mask_sparsity",
        "weight_sparsity",
        "last_coactivations",
    ]

    def is_sparse(self, module):
        if isinstance(module, DSConv2d):
            return "sparse_conv"

    def setup(self):
        super().setup()
        # find sparse layers
        self.sparse_conv_modules = []
        for m in list(self.network.modules()):
            if self.is_sparse(m):
                self.sparse_conv_modules.append(m)
        # print(self.sparse_conv_modules)

    def _post_epoch_updates(self, dataset=None):
        """
        Only change in the model is here.
        In order to work, need to use networks which have DSConv2d layer
        which network is being used?
        """
        print("calling post epoch")
        super()._post_epoch_updates(dataset)

        # go through named modules
        for idx, module in enumerate(self.sparse_conv_modules):
            # if it is a dsconv layer
            # print("layer type: ", module.__class__)
            # print(isinstance(module, DSConv2d))
            # print("progressing connections")
            # Log coactivation before pruning - otherwise they get reset.
            self.log["hist_" + "coactivations_" + str(idx)] = module.coactivations
            # Prune. Then log some params.
            module.progress_connections()
            print("progressing")
            for attr in self.log_attrs:
                value = getattr(module, attr) if hasattr(module, attr) else -2
                if isinstance(value, Iterable):
                    attr = "hist_" + attr
                self.log[attr + "_" + str(idx)] = value

            if isinstance(module, (DSConv2d, SparseConv2d)):
                self.log["sparsity_" + str(idx)] = calc_sparsity(module.weight)


class DSNNConvOnlyHeb(BaseModel):
    """
    Similar to other sparse models, but the focus here is on convolutional layers as
    opposed to dense layers.
    """

    log_attrs = [
        "pruning_iterations",
        "kept_frac",
        "prune_mask_sparsity",
        "keep_mask_sparsity",
        "weight_sparsity",
        "last_coactivations",
    ]

    def is_sparse(self, module):
        if isinstance(module, DSConv2d):
            return "sparse_conv"

    def setup(self):
        super().setup()
        # find sparse layers
        self.sparse_conv_modules = []
        for m in list(self.network.modules()):
            if self.is_sparse(m):
                self.sparse_conv_modules.append(m)
        # print(self.sparse_conv_modules)

    def _post_epoch_updates(self, dataset=None):
        """
        Only change in the model is here.
        In order to work, need to use networks which have DSConv2d layer
        which network is being used?
        """
        print("calling post epoch")
        super()._post_epoch_updates(dataset)

        # go through named modules
        for idx, module in enumerate(self.sparse_conv_modules):
            # if it is a dsconv layer
            # print("layer type: ", module.__class__)
            # print(isinstance(module, DSConv2d))
            # print("progressing connections")
            # Log coactivation before pruning - otherwise they get reset.
            self.log["hist_" + "coactivations_" + str(idx)] = module.coactivations
            # Prune. Then log some params.
            module.progress_connections()
            print("progressing")
            for attr in self.log_attrs:
                value = getattr(module, attr) if hasattr(module, attr) else -2
                if isinstance(value, Iterable):
                    attr = "hist_" + attr
                self.log[attr + "_" + str(idx)] = value

            if isinstance(module, (DSConv2d, SparseConv2d)):
                self.log["sparsity_" + str(idx)] = calc_sparsity(module.weight)
