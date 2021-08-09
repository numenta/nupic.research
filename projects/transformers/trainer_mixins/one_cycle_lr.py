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
import warnings

from torch.optim.lr_scheduler import OneCycleLR


class OneCycleLRMixin:
    """
    Mixin to HF Trainer for using the `OneCycleLR`_. See documentation for argument
    details. Arguments for the one-cycle lr schedule must be passed though
    'trainer_mixin_args'.
    When `Deepspeed`_ is enabled, this mixin will convert the parameters from
    pytorch's `OneCycleLR`_ implementation to `Deepspeed`_'.

    .. _OneCycleLR:
        https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    .. _Deepspeed:
        https://deepspeed.readthedocs.io/en/latest/schedulers.html#onecycle
    """
    def __init__(self, model, args, **kwargs):
        if args.deepspeed is not None:
            self._configure_deepspeed(args)

        super().__init__(model=model, args=args, **kwargs)

    def _configure_deepspeed(self, training_args):
        """
        Configure deepspeed OneCycleLR scheduler by converting the original
        pytorch parameters passed via 'trainer_minix_args'.

       .. note::
            Pytorch uses one cycle with two phases:
                step up followed by a step down phase.
            Deepspeed uses one cycle with three phases:
                step up followed by a step down, and a post cycle decay phase.
            Due to the different implementations this conversion is not an exact
            match.
        """
        mixin_args = training_args.trainer_mixin_args
        pct_start = mixin_args.get("pct_start", 0.3)
        cycle_momentum = mixin_args.get("cycle_momentum", True)
        base_momentum = mixin_args.get("base_momentum", 0.85)
        max_momentum = mixin_args.get("max_momentum", 0.95)
        div_factor = mixin_args.get("div_factor", 25)
        final_div_factor = mixin_args.get("final_div_factor", 1e4)
        last_epoch = mixin_args.get("last_epoch", -1)
        max_lr = mixin_args.get("max_lr", 1e-2)
        anneal_strategy = mixin_args.get("anneal_strategy", "linear")
        if anneal_strategy != "linear":
            warnings.warn(
                f"Deepspeed does not support {anneal_strategy} anneal_strategy")

        max_steps = training_args.max_steps
        initial_lr = max_lr / div_factor
        min_lr = initial_lr / final_div_factor
        first_step_size = int(max_steps * pct_start)
        remaining_steps = max_steps - first_step_size
        second_step_ratio = (max_lr - min_lr) / remaining_steps
        second_step_size = int((max_lr - initial_lr) / second_step_ratio)
        decay_step_size = int((initial_lr - min_lr) / second_step_ratio) + 1
        decay_lr_rate = 1

        # Update deepspeed LR scheduler configuration
        hf_deepspeed_config = training_args.hf_deepspeed_config
        config = hf_deepspeed_config.config
        scheduler = config.setdefault("scheduler", {})
        scheduler_type = scheduler.setdefault("type", "OneCycle")
        assert scheduler_type == "OneCycle"

        params = scheduler.setdefault("params", {})
        params.setdefault("cycle_first_step_size", first_step_size)
        params.setdefault("cycle_second_step_size", second_step_size)
        params.setdefault("decay_step_size", decay_step_size)
        params.setdefault("cycle_min_lr", initial_lr)
        params.setdefault("cycle_max_lr", max_lr)
        params.setdefault("decay_lr_rate", decay_lr_rate)
        params.setdefault("cycle_min_mom", base_momentum)
        params.setdefault("cycle_max_mom", max_momentum)
        params.setdefault("cycle_momentum", cycle_momentum)
        params.setdefault("last_batch_iteration", last_epoch)

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler. This overrides super
        in a way that just customizes the lr scheduler while the optimizer remains the
        default.
        .. note::
            Deepspeed has its own implementation of OneCycleLR and huggingface
            won't call this method when deepspeed is enabled.
        """

        # Unpack arguments from trainer_mixin_args
        mixin_args = self.args.trainer_mixin_args

        max_lr = mixin_args.get("max_lr", 1e-2)
        pct_start = mixin_args.get("pct_start", 0.3)
        anneal_strategy = mixin_args.get("anneal_strategy", "linear")
        cycle_momentum = mixin_args.get("cycle_momentum", True)
        base_momentum = mixin_args.get("base_momentum", 0.85)
        max_momentum = mixin_args.get("max_momentum", 0.95)
        div_factor = mixin_args.get("div_factor", 25)
        final_div_factor = mixin_args.get("final_div_factor", 1e4)
        last_epoch = mixin_args.get("last_epoch", -1)

        # Now define the lr scheduler, given the optimizer.
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            total_steps=num_training_steps,
            max_lr=max_lr,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            last_epoch=last_epoch,
        )
