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
import wandb
from pandas import DataFrame
from torch import count_nonzero
from transformers import TrainerCallback

from nupic.research.frameworks.pytorch.model_utils import (
    count_nonzero_params,
    filter_modules,
)
from nupic.torch.modules.sparse_weights import SparseWeightsBase, rezero_weights

__all__ = [
    "RezeroWeightsCallback",
    "PlotDensitiesCallback",
    "plot_density_per_layer",
    "plot_density_delta",
    "calc_sparsity",
    "calc_model_sparsity",
    "get_density_by_layer"
]


class RezeroWeightsCallback(TrainerCallback):
    """
    This rezeros the weights of a sparse model after each iteration and logs the
    sparsity of the BERT model and its encoder.

    :param log_steps: how often to log the sparsity of the model
    """

    def __init__(self, log_steps=1000):
        self.log_steps = log_steps

    def on_init_end(self, args, state, control, model, **kwargs):
        """Log sparsity of the model and the sparsity of just the encoder."""

        model.apply(rezero_weights)

        num_total, num_nonzero = count_nonzero_params(model)
        model_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"Non-zero Params / Total Params, {num_nonzero:,} / {num_total:,}")
        logging.info(f"   Model Sparsity={model_sparsity:.4f}")

        num_total, num_nonzero = count_nonzero_params(model.bert.encoder)
        encoder_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"   Encoder Sparsity={encoder_sparsity:0.4f}")

        num_total, num_nonzero = count_nonzero_params(model.bert)
        bert_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"   Bert Sparsity={bert_sparsity:0.4f}")

        if wandb.run is not None:
            wandb.run.summary.update(dict(
                bert_on_params_at_init=num_nonzero,
                bert_sparsity_at_init=bert_sparsity,
            ))

    def on_train_end(self, args, state, control, model, **kwargs):

        num_total, num_nonzero = count_nonzero_params(model.bert)
        bert_sparsity = 1 - (num_nonzero / num_total)
        logging.info(f"   Bert Sparsity={bert_sparsity:0.4f}")

        if wandb.run is not None:
            wandb.run.summary.update(dict(
                bert_on_params_at_end=num_nonzero,
                bert_sparsity_at_end=bert_sparsity,
            ))

    def on_step_end(self, args, state, control, model, **kwargs):
        """Rezero weights and log sparsity."""

        model.apply(rezero_weights)

        # Log sparsity to wandb
        if wandb.run is not None and state.global_step % self.log_steps == 0:
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

            wandb.log(logs, commit=False)
            control.should_log = True

        return control


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
        df_density_by_layer = get_density_by_layer(self.sparse_modules)
        density_per_layer = plot_density_per_layer(
            df_density_by_layer,
            self.initial_density
        )
        wandb.log({"density_per_layer": wandb.Image(density_per_layer)}, commit=False)

        # Plot plot change in on params for each layer.
        df_delta_on_params = get_delta_on_params(
            self.sparse_modules,
            self.initial_on_params
        )
        density_delta = plot_density_delta(df_delta_on_params)
        wandb.log({"delta_on_params": wandb.Image(density_delta)}, commit=False)

        control.should_log = True
        return control


# -------------
# Utilities
# -------------

def plot_density_per_layer(df_density_by_layer, initial_density=None):

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.stripplot(
        data=df_density_by_layer,
        y="density",
        x="layer",
        hue=None,
        color="firebrick",
        ax=ax,
    )
    if initial_density is not None:
        ax.axhline(initial_density, label="Initial Density")
        ax.legend(loc="lower center", bbox_to_anchor=(0.2, 0), ncol=2)
        ax.set_title("Density Per Layer")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=-45,
            ha="left",
            rotation_mode="anchor"
        )

    return ax


def plot_density_delta(df_delta_on_params):
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
    return ax


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


def get_density_by_layer(sparse_modules):
    """
    This creates a dataframe with entries (layer name, density of layer).
    """
    df = DataFrame(columns=["layer", "density"])
    for n, m in sparse_modules.items():

        subname = ".".join(n.split(".")[3:])
        layer_name = f"{subname} {tuple(m.weight.shape)}"
        density = 1 - calc_sparsity(m.weight)
        df.loc[len(df.index)] = (layer_name, density.item())

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
        df.loc[len(df.index)] = (layer_name, delta.item())

    return df
