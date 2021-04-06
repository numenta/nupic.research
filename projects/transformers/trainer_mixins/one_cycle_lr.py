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
    details.

    .. _OneCycleLR:
        https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.OneCycleLR
    """

    def __init__(
        self,
        max_lr=1e-2,
        pct_start=0.3,
        anneal_strategy="linear",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=1e4,
        last_epoch=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_lr = max_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.last_epoch = last_epoch

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler. This overrides super
        in a way that just customizes the lr scheduler while the optimizer remains the
        default.
        """

        # Set lr scheduler to dummy variable so it's not created in the call to super.
        self.lr_scheduler = ...

        # Create just the optimizer.
        super().create_optimizer_and_scheduler(num_training_steps)

        # Now define the lr scheduler, given the optimizer.
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            total_steps=num_training_steps,
            max_lr=self.max_lr,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            cycle_momentum=self.cycle_momentum,
            base_momentum=self.base_momentum,
            max_momentum=self.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            last_epoch=self.last_epoch,
        )
