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
import os
from collections import MutableMapping

import wandb
from ray.tune.integration.wandb import WandbLoggerCallback
from transformers.integrations import (
    INTEGRATION_TO_CALLBACK,
    WandbCallback,
    is_wandb_available,
    logger,
)


"""
This file serves to extend the functionalities of automated integrations such as those
for wandb.
"""


class CustomWandbCallback(WandbCallback):
    """
    Customized integration with Wandb. Added features include
        - logging of eval metrics to run summary
        - class method (early_init) to manually initialize Wandb;
          ideally earlier in the setup to include more logs

    This code is adapted from
        https://github.com/huggingface/transformers/blob/1438c487df5ce38a7b2ae30877b3074b96a423dd/src/transformers/integrations.py#L575

    See the Transformers Wandb integration for more details
        https://huggingface.co/transformers/master/main_classes/callback.html?highlight=wandb#transformers.integrations.WandbCallback
    """

    @classmethod
    def early_init(cls, trainer_args, local_rank):
        has_wandb = is_wandb_available()
        assert has_wandb, \
            "WandbCallback requires wandb to be installed. Run `pip install wandb`."

        logger.info("Initializing wandb on rank", local_rank)
        if local_rank not in [-1, 0]:
            return

        # Deduce run name and group.
        init_args = {}
        if hasattr(trainer_args, "trial_name") and trainer_args.trial_name is not None:
            run_name = trainer_args.trial_name
            init_args["group"] = trainer_args.run_name
        else:
            run_name = trainer_args.run_name

        wandb.init(
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            name=run_name,
            reinit=True,
            **init_args
        )

        return wandb.run.id

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.
        """
        super().setup(args, state, model, **kwargs)

        if state.is_world_process_zero and self._wandb is not None:

            # Log the mixin args to the wandb config.
            if hasattr(args, "trainer_mixin_args"):
                flattened_args = flatten_dict(
                    args.trainer_mixin_args,
                    parent_key="trainer_mixin_args"
                )
                wandb.config.update(flattened_args)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Add the following logs to the run summary
            - eval loss
            - eval perplexity
            - the train runtime in hours
        """

        super().on_evaluate(args, state, control, metrics=None, **kwargs)

        if metrics is None or wandb.run is None:
            return

        run = wandb.run
        summary = run.summary

        # Log eval loss and perplexity.
        if "eval_loss" in metrics:

            eval_loss = metrics["eval_loss"]
            perplexity = math.exp(eval_loss)

            eval_results = {
                "eval/perplexity": perplexity,
                "eval/loss": eval_loss,
            }
            summary.update(eval_results)
            wandb.log(eval_results, commit=False)


# Update the integrations. By updating this dict, any custom integration
# will be used automatically by the Trainer.
INTEGRATION_TO_CALLBACK.update({
    "wandb": CustomWandbCallback,
})


# -----
# Utils
# -----


def flatten_dict(d, parent_key="", seperator="."):
    """
    Flatten a (possibly) nested set of dictionaries.
    """

    # Collect the flattened items as a list of tuples.
    items = []
    for k, v in d.items():

        # The parent key will be '' at the first level but non null for the rest.
        new_key = parent_key + seperator + k if parent_key else k

        # Recurse for dictionary like objects.
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, seperator=seperator).items())
        # Otherwise append as is (e.g. for list and strings)
        else:
            items.append((new_key, v))

    return dict(items)


def init_ray_wandb_logger_callback(training_args):
    """
    Initialize the ray wandb integration, used specifically for hyperparameter
    tuning. Returns either None or a list containing the initialized callback,
    so the output can be passed directly to hp_search_kwargs.
    """
    has_wandb = is_wandb_available()
    if not has_wandb:
        return None

    project = os.getenv("WANDB_PROJECT", "huggingface")
    group = training_args.run_name
    callbacks = [WandbLoggerCallback(
        project=project,
        group=group,
    )]

    return callbacks
