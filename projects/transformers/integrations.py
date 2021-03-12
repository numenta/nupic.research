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

import math

import wandb
from transformers.integrations import INTEGRATION_TO_CALLBACK, WandbCallback


"""
This file serves to extend the functionalities of automated integrations such as those
for wandb.
"""


class CustomWandbCallback(WandbCallback):
    """
    Log perplexity to wandb following evaluation.
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        super().on_evaluate(args, state, control, metrics=None, **kwargs)

        if metrics is None:
            return

        perplexity = math.exp(metrics["eval_loss"])
        print("CustomWandbCallback: Logging perplexity to wandb run summary.")
        if wandb.run is not None:
            wandb.run.summary["eval/perplexity"] = perplexity


# Update the integrations. These will be used automatically.
INTEGRATION_TO_CALLBACK.update({
    "wandb": CustomWandbCallback,
})
