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

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from pandas import DataFrame
from torch import count_nonzero
from transformers import TrainerCallback

from nupic.research.frameworks.dynamic_sparse import (
    CosineDecayPruneScheduler,
    global_add_by_abs_grad,
    global_prune_by_abs_weight,
)
from nupic.research.frameworks.pytorch.model_utils import (
    count_nonzero_params,
    filter_modules,
)
from nupic.torch.modules.sparse_weights import SparseWeightsBase, rezero_weights

__all__ = [
    "RezeroWeightsCallback",
    "RigLCallback",
    "PlotDensitiesCallback"
]


class RezeroWeightsCallback(TrainerCallback):
    """
    This rezeros the weights of a sparse model after each iteration and logs the
    sparsity of the BERT model and its encoder.
    """

    def on_init_end(self, args, state, control, model, **kwargs):
        """Log sparsity of the model and the sparsity of just the encoder."""

        model.apply(rezero_weights)

        num_total, num_nonzero = count_nonzero_params(model)
        model_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"Non-zero Params / Total Params, {num_nonzero:,} / {num_total:,}")
        logging.info(f"   Model Sparsity={model_sparsity:.4f}")

        num_total, num_nonzero = count_nonzero_params(model.bert)
        bert_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"   Bert Sparsity={bert_sparsity:0.4f}")

        num_total, num_nonzero = count_nonzero_params(model.bert.encoder)
        encoder_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"   Encoder Sparsity={encoder_sparsity:0.4f}")

    def on_step_end(self, args, state, control, model, **kwargs):
        """Rezero weights and log sparsity."""

        model.apply(rezero_weights)

        # Log sparsity to wandb
        if wandb.run is not None:
            num_total, num_nonzero = count_nonzero_params(model)
            model_sparsity = 1 - (num_nonzero / num_total)

            num_total, num_nonzero = count_nonzero_params(model.bert)
            bert_sparsity = 1 - (num_nonzero / num_total)

            num_total, num_nonzero = count_nonzero_params(model.bert.encoder)
            encoder_sparsity = 1 - (num_nonzero / num_total)

            logs = dict(
                model_sparsity=model_sparsity,
                bert_sparsity=bert_sparsity,
                encoder_sparsity=encoder_sparsity
            )

            wandb.log(logs, step=state.global_step)


class RigLCallback(TrainerCallback):
    """
    Perform `RigL`_ dynamic sparsity every N steps on the sparse modules within the
    network. Weights are pruned according to their ranks absolute values and regrowing
    (adding weights) occurs by ranking their absolute gradients. As per the original
    paper, the pruning decreases over time according to a cosine decay.

    .. _RigL: https://arxiv.org/abs/1911.11134

    :param prune_fraction: initial fraction of weights to prune
    :param prune_freq: how often, in iterations, to prune and regrow weights
    :param warmup_steps: defaults to prune_freq; e.g. prune_freq of 100 will allow 100
                         steps before pruning for the first time
    """

    def __init__(
        self,
        prune_fraction=0.3,
        prune_freq=100,
        warmup_steps=None
    ):
        self.prune_fraction = prune_fraction
        self.prune_freq = prune_freq
        self.prune_scheduler = None
        self.warmup_steps = warmup_steps

    def on_init_end(self, args, state, control, model, optimizer=None, **kwargs):
        """Save a list of the sparse modules and initialize the pruning schedule"""

        warmup_steps = self.warmup_steps or self.prune_freq
        self.prune_scheduler = CosineDecayPruneScheduler(
            total_steps=args.max_steps,
            prune_fraction=self.prune_fraction,
            warmup_steps=warmup_steps - 1
        )
        self.sparse_modules = filter_modules(
            model, include_modules=[SparseWeightsBase]
        ).values()

    def on_step_end(
        self, args, state, control, model=None,
        train_dataloader=None, optimizer=None, **kwargs
    ):
        """Prune and regrow weights every 'prune_freq' iterations."""

        if state.global_step % self.prune_freq != 0:
            self.prune_scheduler.step()
            return

        # Pre-prune sparsities.
        param_sparsity0, mask_sparsity0 = calc_cumulative_sparsity(self.sparse_modules)

        # Prune weights.
        model.apply(rezero_weights)
        prune_fraction = self.prune_scheduler.get_prune_fraction()
        num_removed = global_prune_by_abs_weight(self.sparse_modules, prune_fraction)
        model.apply(rezero_weights)

        # Post-prune sparsities.
        param_sparsity1, mask_sparsity1 = calc_cumulative_sparsity(self.sparse_modules)

        # Accumulate gradients over one batch.
        optimizer.zero_grad()
        train_batch = next(iter(train_dataloader))
        inputs_to_device(train_batch, device=args.device)
        output = model(**train_batch)
        output.loss.backward()

        # Regrow weights
        num_add = self.prune_scheduler.get_num_add(num_removed)
        global_add_by_abs_grad(self.sparse_modules, num_add)
        self.prune_scheduler.step()

        # Post-grow sparsities.
        param_sparsity2, mask_sparsity2 = calc_cumulative_sparsity(self.sparse_modules)

        # Log pruning stats.
        actual_pruned = param_sparsity1 - param_sparsity0
        actual_pruned_on_params = actual_pruned / (1 - mask_sparsity0)

        logging.info(f"RigLCallback:")
        logging.info(f"Target: remove {prune_fraction} frac of on params")
        logging.info(f"Actual: removed {actual_pruned_on_params} fraction of on params")

        # For now, the logs are very robust to ensure pruning occurs as expected.
        # TODO: Remove non-essential logging.
        logs = dict({
            "rigl/target_pruned_on_params": prune_fraction,
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
            wandb.log(logs, step=state.global_step)


class PlotDensitiesCallback(TrainerCallback):
    """
    Callback to plot the densities of each layer as well as well as the deltas in the
    numbers of their on-parameters. This is particularly useful for training dynamic
    sparse networks where the densities of each layer may change over time.

    :param plot_freq: how often to generate the plots
    """

    def __init__(self, plot_freq=1000):
        self.plot_freq = plot_freq
        self.initial_on_params = dict()  # used to calculate the delta in on-params
        self.sparse_modules = None

    def on_init_end(self, args, state, control, model=None, **kwargs):
        """
        Save a list of all the sparse modules and a dict of their initial densities.
        """
        self.sparse_modules = filter_modules(model, include_modules=[SparseWeightsBase])
        for name, module in self.sparse_modules.items():
            zero_mask = module.zero_mask.bool()
            self.initial_on_params[name] = zero_mask.numel() - count_nonzero(zero_mask)
        initial_sparsity = getattr(args, "config_kwargs", {}).get("sparsity", 0)
        self.initial_density = 1 - initial_sparsity

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Plot the densities of each layer and plot the change in on-params of each layer.
        """

        if state.global_step % self.plot_freq != 0:
            return

        if wandb.run is None:
            return

        # Plot densities for each layer.
        df_dendity_by_layer = get_density_by_layer(self.sparse_modules)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.stripplot(
            data=df_dendity_by_layer,
            y="density",
            x="layer",
            hue=None,
            color="firebrick",
            ax=ax,
        )
        ax.axhline(self.initial_density, label="Initial Density")
        ax.legend(loc="lower center", bbox_to_anchor=(0.2, 0), ncol=2)
        ax.set_title("Density Per Layer")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=-45,
            ha="left",
            rotation_mode="anchor"
        )
        plot = wandb.Image(ax)
        wandb.log({"density_per_layer": plot}, step=state.global_step)

        # Plot plot change in on params for each layer.
        df_delta_on_params = get_delta_on_params(
            self.sparse_modules,
            self.initial_on_params
        )
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.stripplot(
            data=df_delta_on_params,
            y="delta_on_params",
            x="layer",
            hue=None,
            color="firebrick",
            ax=ax,
        )
        ax.axhline(0)
        ax.set_title("Change in On-Params Per Layer")
        ax.set_ylabel("delta on-params")
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=-45,
            ha="left",
            rotation_mode="anchor",
        )
        plot = wandb.Image(ax)
        wandb.log({"delta_on_params": plot}, step=state.global_step)


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


def calc_sparsity(tensor):
    """Calculate the sparsity of a given tensor."""
    num_total = tensor.numel()
    num_zero = num_total - count_nonzero(tensor)
    return num_zero / num_total


def calc_model_sparsity(model):
    """Calculate the sparsity of a given model."""
    tot, nz = count_nonzero_params(model)
    sparsity = 1 - nz / tot
    return sparsity


def calc_cumulative_sparsity(sparse_modules):
    """
    Calculate the sparsities across a list of sparse modules. Both the weight sparsity
    and the zero mask sparsity is calculated.
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


def get_density_by_layer(sparse_modules):
    """
    This creates a dataframe with entries (layer name, density of layer).
    """
    df = DataFrame(columns=["layer", "density"])
    for n, m in sparse_modules.items():

        subname = ".".join(n.split(".")[3:])
        layer_name = f"{subname} {tuple(m.weight.shape)}"
        density = 1 - calc_sparsity(m.weight)
        df.loc[len(df.index)] = (layer_name, density)

    return df


def get_delta_on_params(sparse_modules, initial_on_params):
    """
    This creates a dataframe with entries (layer name, change in num on params).

    The initial_on_params is dict that maps the layer name to their initial number of
    on parameters.
    """
    df = DataFrame(columns=["layer", "delta_on_params"])
    for n, m in sparse_modules.items():

        subname = ".".join(n.split(".")[3:])
        layer_name = f"{subname} {tuple(m.weight.shape)}"

        zero_mask = m.zero_mask.bool()
        on_params = zero_mask.numel() - count_nonzero(zero_mask)
        delta = on_params - initial_on_params[n]
        df.loc[len(df.index)] = (layer_name, delta)

    return df
