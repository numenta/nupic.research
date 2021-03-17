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

import wandb
from transformers import TrainerCallback

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.torch.modules import rezero_weights

__all__ = [
    "RezeroWeightsCallback",
]


class RezeroWeightsCallback(TrainerCallback):

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
