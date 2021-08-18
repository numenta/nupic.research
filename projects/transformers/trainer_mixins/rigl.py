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

import logging

import torch
import wandb
from torch import count_nonzero

from nupic.research.frameworks.dynamic_sparse import (
    CosineDecayPruneScheduler,
    global_add_by_abs_grad,
    global_prune_by_abs_weight,
)
from nupic.research.frameworks.pytorch.model_utils import filter_modules
from nupic.torch.modules.sparse_weights import SparseWeightsBase, rezero_weights


class RigLMixin:
    """
    Perform `RigL`_ dynamic sparsity every N steps on the sparse modules within the
    network. Weights are pruned according to their ranks absolute values and regrowing
    (adding weights) occurs by ranking their absolute gradients. As per the original
    paper, the pruning decreases over time according to a cosine decay.

    .. _RigL: https://arxiv.org/abs/1911.11134

    Params to add to 'trainer_mixin_args':
    :param prune_fraction: initial fraction of weights to prune
    :param prune_freq: how often, in iterations, to prune and regrow weights
    :param warmup_steps: defaults to prune_freq; e.g. prune_freq of 100 will allow 100
                         steps before pruning for the first time
    :param verbose_rigl_logging: defaults to false
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        mixin_args = self.args.trainer_mixin_args

        self.prune_fraction = mixin_args.get("prune_fraction", 0.3)
        self.prune_freq = mixin_args.get("prune_freq", 100)
        self.warmup_steps = mixin_args.get("warmup_steps", self.prune_freq)
        total_steps = self.args.max_steps * self.args.gradient_accumulation_steps
        self.prune_scheduler = CosineDecayPruneScheduler(
            total_steps=total_steps,
            prune_fraction=self.prune_fraction,
            warmup_steps=self.warmup_steps
        )
        self.sparse_modules = None
        self.verbose_rigl_logging = mixin_args.get("verbose_rigl_logging", False)

    def training_step(self, model, inputs):
        """Prune and regrow weights every 'prune_freq' iterations."""

        train_loss = super().training_step(model, inputs)

        if self.state.global_step % self.prune_freq != 0:
            self.prune_scheduler.step()
            return train_loss

        # Retrieve sparse modules (e.g. SparseWeights) after model has been setup for
        # distributed training, if it has.
        if self.sparse_modules is None:
            self.sparse_modules = filter_modules(
                model, include_modules=[SparseWeightsBase]
            ).values()
        sparse_modules = self.sparse_modules

        # Pre-prune sparsities (for verbose logging).
        model.apply(rezero_weights)
        if self.verbose_rigl_logging:
            param_sparsity0, mask_sparsity0 = calc_cumulative_sparsity(sparse_modules)

        # If prune fraction is 0, say for a warmup step, return and don't prune.
        prune_fraction = self.prune_scheduler.get_prune_fraction()
        if prune_fraction == 0:
            self.prune_scheduler.step()
            return train_loss

        # Prune weights.
        num_removed = global_prune_by_abs_weight(self.sparse_modules, prune_fraction)
        model.apply(rezero_weights)

        # Post-prune sparsities (for verbose logging).
        if self.verbose_rigl_logging:
            param_sparsity1, mask_sparsity1 = calc_cumulative_sparsity(sparse_modules)

        # Accumulate gradients over one batch.
        self.optimizer.zero_grad()
        train_dataloader = self.callback_handler.train_dataloader
        train_batch = next(iter(train_dataloader))
        inputs_to_device(train_batch, device=self.args.device)
        batch_loss = self.compute_loss(model, train_batch)
        batch_loss.backward()

        # Regrow weights
        num_add = self.prune_scheduler.get_num_add(num_removed)
        global_add_by_abs_grad(self.sparse_modules, num_add)
        self.prune_scheduler.step()

        logs = dict({
            "rigl/target_pruned_on_params": prune_fraction,
        })

        # Post-grow sparsities (for verbose logging).
        if self.verbose_rigl_logging:
            param_sparsity2, mask_sparsity2 = calc_cumulative_sparsity(sparse_modules)

            # Log pruning stats.
            actual_pruned = param_sparsity1 - param_sparsity0
            actual_pruned_on_params = actual_pruned / (1 - mask_sparsity0)

            logging.debug(f"Target: remove {prune_fraction} frac of on params")
            logging.debug(f"Actual: removed {actual_pruned_on_params} "
                          "fraction of on params")

            # These are logs are very robust to ensure the actual percentage and count
            # of pruned-params match the target amounts.
            logs = dict({
                "rigl/actual_pruned_on_params": actual_pruned_on_params,
                "rigl/target_pruned_all_params": prune_fraction * mask_sparsity0,
                "rigl/actual_pruned_all_params": actual_pruned,
                "rigl/pre_prune_param_sparsity": param_sparsity0,
                "rigl/pre_prune_mask_sparsity": mask_sparsity0,
                "rigl/post_prune_param_sparsity": param_sparsity1,
                "rigl/post_prune_mask_sparsity": mask_sparsity1,
                "rigl/pre_grow_param_sparsity": param_sparsity2,
                "rigl/post_grow_mask_sparsity": mask_sparsity2,
            })

        if wandb.run is not None:
            wandb.log(logs, commit=False)

        return train_loss

# -------------
# Utilities
# -------------


def inputs_to_device(inputs, device):
    """
    Prep inputs dict by transferring all inputs to the device.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)


def calc_cumulative_sparsity(sparse_modules):
    """
    Calculate the sparsities across a list of sparse modules. Both the weight sparsity
    and the zero mask sparsity are calculated.
    """
    total_off = 0
    total_zero = 0
    total_params = 0
    for m in sparse_modules:
        total_off += count_nonzero(m.zero_mask)
        total_zero += m.weight.numel() - count_nonzero(m.weight)
        total_params += m.weight.numel()

    mask_sparsity = total_off / total_params
    param_sparsity = total_zero / total_params
    return param_sparsity, mask_sparsity
