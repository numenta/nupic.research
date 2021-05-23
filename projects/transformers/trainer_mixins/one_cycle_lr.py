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

from torch.optim.lr_scheduler import OneCycleLR


class OneCycleLRMixin:
    """
    Mixin to HF Trainer for using the `OneCycleLR`_. See documentation for argument
    details. Arguments for the one-cycle lr schedule must be passed though
    'trainer_mixin_args'.

    .. _OneCycleLR:
        https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    """

    def create_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler. This overrides super
        in a way that just customizes the lr scheduler while the optimizer remains the
        default.
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
