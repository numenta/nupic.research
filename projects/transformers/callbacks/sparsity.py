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

import wandb
from transformers import TrainerCallback
from transformers.utils import logging

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.torch.modules import SparseWeights, rezero_weights

logger = logging.get_logger(__name__)

__all__ = [
    "RezeroWeightsCallback",
    "SparsifyFCLayersCallback",
]


class SparsifyFCLayersCallback(TrainerCallback):
    """
    Sparsifies the hidden and output fully connected layer for a
    BERT HuggingFace model
    """

    def __init__(self, sparsity: float = 0.5, num_sparse_layers: int = 12):
        self.sparsity = sparsity
        self.num_sparse_layers = num_sparse_layers

    def on_init_end(self, args, state, control, model, **kwargs):
        """Replace linear layers with sparse layers"""
        device = model.device
        for idx in range(self.num_sparse_layers):
            intermediate_layer = model.bert.encoder.layer[idx].intermediate.dense
            model.bert.encoder.layer[idx].intermediate.dense = \
                SparseWeights(intermediate_layer, sparsity=self.sparsity).to(device)

            output_layer = model.bert.encoder.layer[idx].output.dense
            model.bert.encoder.layer[idx].output.dense = \
                SparseWeights(output_layer, sparsity=self.sparsity).to(device)

    def on_step_end(self, args, state, control, model, **kwargs):
        """Rezero weights"""
        model.apply(rezero_weights)


class RezeroWeightsCallback(TrainerCallback):

    def on_init_end(self, args, state, control, model, **kwargs):
        model.apply(rezero_weights)

        num_tol, num_nonzero = count_nonzero_params(model)
        model_sparsity = 1 - (num_nonzero / num_tol)
        logger.info(f"Non-zero Params / Total Params, {num_nonzero:,} / {num_tol:,}")
        logger.info(f"   Model Sparsity={model_sparsity:.4f}")

        num_tol, num_nonzero = count_nonzero_params(model.bert.encoder)
        encoder_sparsity = 1 - (num_nonzero / num_tol)
        logger.info(f"   Encoder Sparsity={encoder_sparsity:0.4f}")

    def on_step_end(self, args, state, control, model, **kwargs):
        """Rezero weights and log sparsity."""

        model.apply(rezero_weights)

        # Log sparsity to wandb
        if wandb.run is not None:
            num_tol, num_nonzero = count_nonzero_params(model)
            model_sparsity = 1 - (num_nonzero / num_tol)

            num_tol, num_nonzero = count_nonzero_params(model.bert.encoder)
            encoder_sparsity = 1 - (num_nonzero / num_tol)

            logs = dict(
                model_sparsity=model_sparsity,
                encoder_sparsity=encoder_sparsity
            )

            wandb.log(logs, step=state.global_step)
