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
            pruning_es=True,
            pruning_es_patience=0,
            pruning_es_window_size=6,
            pruning_es_threshold=0.02,
            pruning_interval=1,
            hebbian_prune_perc=0,
            hebbian_grow=True,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # initialize hebbian learning
        self.network.hebbian_learning = True
        # count number of cycles, compare with patience
        self.pruning_es_cycles = 0
        self.last_survival_ratios = deque(maxlen=self.pruning_es_window_size)

    def _post_epoch_updates(self, dataset=None):
        super()._post_epoch_updates(dataset)
        # zero out correlations
        self._reinitialize_weights()
        self.network.correlations = []
        # dynamically decide pruning interval

    def _reinitialize_weights(self):
        """Reinitialize weights."""
        # only run if still learning and if at the right interval
        # current epoch is 1-based indexed
        if self.pruning_active and (self.current_epoch % self.pruning_interval) == 0:
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
                            # log if in debug sparse mode
                            if self.debug_sparse:
                                self.log[
                                    "surviving_synapses_l" + str(idx)
                                ] = survival_ratio

                    # keep track of new synapses to count surviving on next round
                    self.added_synapses[idx] = new_synapses

            # early stop (alternative - keep a moving average)
            # ignore the last layer for now
            mean_survival_ratio = np.mean(survival_ratios[:-1])
            if not np.isnan(mean_survival_ratio):
                self.last_survival_ratios.append(mean_survival_ratio)
                if self.debug_sparse:
                    self.log["surviving_synapses_avg"] = mean_survival_ratio
                if self.pruning_es:
                    ma_survival = (
                        np.sum(list(self.last_survival_ratios))
                        / self.pruning_es_window_size
                    )
                    if ma_survival < self.pruning_es_threshold:
                        self.pruning_es_cycles += 1
                        self.last_survival_ratios.clear()
                    if self.pruning_es_cycles > self.pruning_es_patience:
                        self.pruning_active = False

            # keep track of mask sizes when debugging
            if self.debug_sparse:
                for idx, m in enumerate(self.masks):
                    self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()

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
            self.log["weight_keep_mask_l" + str(idx)] = torch.sum(
                weight_keep_mask
            ).item()

            # no gradient mask, just a keep mask
            keep_mask = weight_keep_mask

            # calculate number of parameters to add
            num_add = num_params - torch.sum(keep_mask).item()
            self.log["missing_weights_l" + str(idx)] = num_add
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
            self.log["added_synapses_l" + str(idx)] = torch.sum(add_mask).item()

        # track added connections
        return new_mask, keep_mask, add_mask


class DSNNMixedHeb(DSNNHeb):
    """Improved results compared to DSNNHeb"""

    def prune(self, weight, num_params, corr, idx=0):
        """
        Grow by correlation
        Prune by magnitude
        """
        with torch.no_grad():

            # print("corr dimension", corr.shape)
            # print("weight dimension", weight.shape)

            # transpose to fit the weights
            corr = corr.t()
            shape = np.prod(weight.shape)

            tau = self.hebbian_prune_perc
            # decide which weights to remove based on correlation
            kth = int((1 - tau) * np.prod(corr.shape))
            corr_threshold, _ = torch.kthvalue(corr.contiguous().view(-1), kth)
            hebbian_keep_mask = (corr > corr_threshold).to(self.device)

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
            keep_mask = weight_keep_mask & hebbian_keep_mask
            self.log["weight_keep_mask_l" + str(idx)] = (
                torch.sum(keep_mask).item() / shape
            )

            # calculate number of parameters to add
            num_add = max(
                num_params - torch.sum(keep_mask).item(), 0
            )  # TODO: debug why < 0
            self.log["missing_weights_l" + str(idx)] = num_add / shape

            # added option to have hebbian grow or not
            if self.hebbian_grow:
                # remove the ones which will already be kept
                corr *= (keep_mask == 0).float()
                # get kth value, based on how many weights to add, and calculate mask
                kth = int(np.prod(corr.shape) - num_add)
                # contiguous()
                corr_threshold, _ = torch.kthvalue(corr.contiguous().view(-1), kth)
                add_mask = (corr > corr_threshold).to(self.device)
            else:
                current_sparsity = torch.sum(weight == 0).item()
                p_add = num_add / max(current_sparsity, num_add)  # avoid div by zero
                probs = torch.rand(weight.shape).to(self.device) < p_add
                add_mask = probs & (weight == 0)

            new_mask = keep_mask | add_mask
            self.log["added_synapses_l" + str(idx)] = torch.sum(add_mask).item() / shape

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
