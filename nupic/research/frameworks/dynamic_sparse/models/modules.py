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

from itertools import product

import numpy as np
import torch

from nupic.research.frameworks.dynamic_sparse.networks import DynamicSparseBase
from nupic.research.frameworks.dynamic_sparse.networks.layers import (
    init_coactivation_tracking,
)


class SparseModule:
    """Module wrapper for sparse layers
    Attributes:
        m (torch.nn.module): Original module to be wrapped.
        pos (int): Position in the list of sparse modules, used for logging
        on_perc (float): Percentage of non-zero weights
        shape (tuple): Shape of the underlying weight matrix
        num_params (int): Number of non-zero params at initialization/
            before training
        mask (tensor): Binary tensor. 1 where connection is active, 0 otherwise
        weight_prune (float): Percentage to be pruned based on magnitude
        hebbian_prune (float): Percentage to be pruned based on hebbian stats
        device (torch.device): Device where to save and compute data.
        added_synapses (tensor): Tensor with synapses added at last growth round.
            Required to log custom metrics.
        last_gradients (tensor): Tensor with gradients at last iteration.
            Required to log custom metrics.
    """

    def __init__(
        self,
        m,
        pos=None,
        on_perc=None,
        num_params=None,
        mask=None,
        weight_prune=None,
        hebbian_prune=None,
        device=None,
    ):
        """document attributes"""
        self.m = m
        self.shape = m.weight.shape
        self.pos = pos
        self.on_perc = on_perc
        self.num_params = num_params
        self.mask = mask
        self.hebbian_prune = hebbian_prune
        self.weight_prune = weight_prune
        self.device = device
        # logging
        self.added_synapses = None
        self.last_gradients = None

    def __repr__(self):
        return str(
            {
                "name": self.m._get_name(),
                "index": self.pos,
                "shape": self.shape,
                "on_perc": self.on_perc,
                "hebbian_prune": self.hebbian_prune,
                "weight_prune": self.weight_prune,
                "num_params": self.num_params,
            }
        )

    def nonzero_params(self):
        return torch.sum(self.mask).item()

    def save_num_params(self):
        self.num_params = self.nonzero_params()

    def init_coactivation_tracking(self):
        if isinstance(self.m, DynamicSparseBase):
            self.m.apply(init_coactivation_tracking)
        else:
            self.m.coactivations = torch.zeros(self.m.weight.shape).to(self.device)

    def reset_coactivations(self, reset=True):
        if isinstance(self.m, DynamicSparseBase):
            self.m.reset_coactivations()
        else:
            self.m.coactivations[:] = 0

    def apply_mask(self):
        if self.mask is not None:
            self.m.weight.data *= self.mask

    def create_mask(self, sparse_type):
        if sparse_type == "precise":
            self._mask_fixed()
        elif sparse_type == "precise_per_output":
            self._mask_fixed_per_output()
        elif sparse_type == "approximate":
            self._mask_stochastic()
        else:
            raise ValueError(
                "{} is an invalid option for sparse type. ".format(sparse_type)
            )

    def get_coactivations(self):
        return self.m.coactivations

    def _mask_fixed_per_output(self):
        """
        Similar implementation to how so dense
        Works in any number of dimension, considering the 1st one is the output
        """
        output_size = self.shape[0]
        input_size = np.prod(self.shape[1:])
        num_add = int(self.on_perc * input_size)
        mask = torch.zeros(self.shape, dtype=torch.bool)
        # loop over outputs
        for dim in range(output_size):
            # select a subsample of the indexes in the output unit
            all_idxs = np.array(list(product(*[range(s) for s in self.shape[1:]])))
            sampled_idxs = np.random.choice(
                range(len(all_idxs)), num_add, replace=False
            )
            selected = all_idxs[sampled_idxs]
            if len(selected) > 0:
                # change from long to wide format
                selected_wide = list(zip(*selected))
                # append the output dimension to wide format
                selected_wide.insert(0, tuple([dim] * len(selected_wide[0])))
                # apply the mask
                mask[selected_wide] = True

        self.mask = mask.float().to(self.device)

    def _mask_stochastic(self):
        """Sthocastic in num of params approach of sparsifying a tensor"""
        self.mask = (torch.rand(self.shape) < self.on_perc).float().to(self.device)

    def _mask_fixed(self):
        """
        Deterministic in number of params approach of sparsifying a tensor
        Sample N from all possible indices
        """
        all_idxs = np.array(list(product(*[range(s) for s in self.shape])))
        num_add = int(self.on_perc * np.prod(self.shape))
        sampled_idxs = np.random.choice(range(len(all_idxs)), num_add, replace=False)
        selected = all_idxs[sampled_idxs]

        mask = torch.zeros(self.shape, dtype=torch.bool)
        if len(selected.shape) > 1:
            mask[list(zip(*selected))] = True
        else:
            mask[selected] = True

        self.mask = mask.float().to(self.device)


class PrunableModule(SparseModule):
    def decay_density(self):
        self.target_density -= self.decay_amount

    def update_mask(self, to_prune):
        """
        Prunes equally positive and negative weights
        to_prune is an upper bound of params being pruned,
        may prune less if not enough positive or negative weights to prune
        """

        weight = self.m.weight.clone().detach()

        # prune positive
        to_prune_pos = int(to_prune / 2)
        weight_pos = weight[weight > 0]
        pos_threshold = 0
        if len(weight_pos) > 0 and to_prune_pos > 0:
            pos_kth = min(to_prune_pos, len(weight_pos))
            if len(weight_pos) > 0:
                pos_threshold, _ = torch.kthvalue(weight_pos, pos_kth)
        prune_pos_mask = (weight <= pos_threshold) & (weight > 0)

        # prune negative
        to_prune_neg = min(torch.sum(prune_pos_mask).item(), to_prune_pos)
        weight_neg = weight[weight < 0]
        neg_threshold = 0
        if len(weight_neg) > 0 and to_prune_neg > 0:
            to_prune_neg = min(to_prune_neg, len(weight_neg))
            neg_kth = len(weight_neg) - to_prune_neg + 1
            if len(weight_neg) > 0:
                neg_threshold, _ = torch.kthvalue(weight_neg, neg_kth)
        prune_neg_mask = (weight >= neg_threshold) & (weight < 0)

        # consolidate
        prune_mask = prune_pos_mask | prune_neg_mask
        new_mask = (weight != 0) & ~prune_mask
        self.mask = new_mask.float().to(self.device)

        # if torch.sum(prune_mask).item() != to_prune:
        #     print("Number of weights pruned doesn't match target density")

    def prune(self):
        """Prune to desired level of sparsity"""

        # calculate number of params to be pruned
        target_params = int(np.prod(self.shape) * self.target_density)
        current_params = torch.sum(self.m.weight != 0).item()
        num_params_to_prune = current_params - target_params
        if num_params_to_prune > 1:
            self.update_mask(num_params_to_prune)
