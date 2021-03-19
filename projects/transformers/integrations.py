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

import wandb
from transformers.integrations import (
    INTEGRATION_TO_CALLBACK,
    WandbCallback,
    is_torch_tpu_available,
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
        assert has_wandb, "WandbCallback requires wandb to be installed. Run `pip install wandb`."

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

    def setup(self, args, state, model, reinit, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.
        """
        if self._wandb is None:
            return

        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                "Automatic Weights & Biases logging enabled, "
                "to disable set os.environ['WANDB_DISABLED'] = 'true'"
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}

            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                run_name = trial_name
                init_args["group"] = args.run_name
            else:
                run_name = args.run_name

            if reinit or wandb.run is None:
                self._wandb.init(
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    config=combined_dict,
                    name=run_name,
                    reinit=reinit,
                    **init_args,
                )
            else:
                wandb.config.update(combined_dict)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps)
                )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Add the following logs to the run summary
            - eval loss
            - eval perplexity
            - the train runtime in hours
        """

        super().on_evaluate(args, state, control, metrics=None, **kwargs)

        if metrics is None:
            return

        eval_loss = metrics["eval_loss"]
        perplexity = math.exp(eval_loss)
        run = wandb.run
        if run is not None:
            run.summary["eval/perplexity"] = perplexity
            run.summary["eval/loss"] = eval_loss

            runtime = run.summary["train/train_runtime"]
            run.summary["train/train_runtime (hrs)"] = runtime / 3600


# Update the integrations. By updating this dict, any custom integration
# will be used automatically by the Trainer.
INTEGRATION_TO_CALLBACK.update({
    "wandb": CustomWandbCallback,
})
