#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import copy
from bisect import bisect

from torch.optim.lr_scheduler import OneCycleLR, _LRScheduler


class ScaledLR(_LRScheduler):
    """
    Multiply the learning rate of each parameter group  by a specific factor
    assigned to the epoch. This LR scheduler could be chained together with
    other schedulers. This is useful when scaling the LR to the batch size.

    .. seealso:: See https://arxiv.org/pdf/1706.02677.pdf

    :param optimizer: Wrapped optimizer
    :param lr_scale: dict mapping initial epoch to LR scale
    :param last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, lr_scale, last_epoch=-1):
        self.lr_scale = lr_scale
        self.epochs = sorted(self.lr_scale.keys())
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        scale = self.lr_scale[self.epochs[bisect(self.epochs, self.last_epoch) - 1]]
        return map(lambda group: group["lr"] * scale, self.optimizer.param_groups)


class ComposedLRScheduler(_LRScheduler):
    """
    Learning scheduler composed of different LR schedulers and optimizer
    parameters to be effective once the number of epochs reaches the specified
    epoch milestone. Similar to :class:`torch.optim.lr_scheduler.MultiStepLR`
    but instead of just updating the LR at the epoch milestone it replaces the
    LR Scheduler and update other optimizer parameters.

    For example::

        # Use "OneCycleLR" for the first 35 epochs and "StepLR" for the rest

        lr_scheduler = ComposedLRScheduler(schedulers={
            0: dict(
                lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
                lr_scheduler_args=dict(
                    max_lr=6.0,
                    div_factor=6,  # initial_lr = 1.0
                    final_div_factor=4000,  # min_lr = 0.00025
                    pct_start=4.0 / 35.0,
                    epochs=35,
                    steps_per_epoch=len(train_loader),
                    anneal_strategy="linear",
                    max_momentum=0.01,
                    cycle_momentum=False,
                ),
                optimizer_args=dict(
                    lr=0.1,
                    weight_decay=0.0001,
                    momentum=0.0,
                    nesterov=False,
                ),
            ),
            35: dict(
                lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
                lr_scheduler_args=dict(
                    gamma=0.1,
                    step_size=10,
                ),
                optimizer_args=dict(
                    lr=0.1,
                    weight_decay=1e-04,
                    momentum=0.9,
                    dampening=0,
                    nesterov=True
                ),
            ),
        })

    :param optimizer:
        Wrapped optimizer
    :type optimizer: torch.optim.optimizer.Optimizer
    :param schedulers:
        dict mapping epoch milestones to LRScheduler and Optimizer parameters
        with the following fields:
        - "optimizer_args": Optimizer arguments to override
        - "lr_scheduler_class": LR Scheduler class
        - "lr_scheduler_args": LR Scheduler class constructor args in addition
                               to optimizer
    :type schedulers: dict[int, dict]
    :param steps_per_epoch: Number of batches/steps per epoch. Must be specified
                            when the LR is updated on every batch. Default 1
    :type steps_per_epoch: int
    :param last_step:
        The index of last step. Default: -1.
    :type last_epoch: int
    """

    def __init__(self, optimizer, schedulers, steps_per_epoch=1, last_epoch=-1):
        self.schedulers = schedulers
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = None
        self.active_milestone = None
        self.milestones = sorted(self.schedulers.keys())
        assert len(self.milestones) > 0

        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def step(self, epoch=None):
        """
        Step should be called after every batch update if OneCycleLR is one of
        the mapped LR Schedulers. Make sure to specify "steps_per_epoch" when
        """
        # Get milestone for current step
        current_step = epoch
        if current_step is None:
            current_step = self.last_epoch + 1
        current_epoch = current_step // self.steps_per_epoch
        current_batch = current_step % self.steps_per_epoch
        current_milestone = self.milestones[bisect(self.milestones, current_epoch) - 1]

        # Update LR scheduler and optimizer once the milestone changes
        if current_batch == 0 and self.active_milestone != current_milestone:
            self.active_milestone = current_milestone
            self._update_optimizer()
            self._update_lr_scheduler()
        elif isinstance(self.lr_scheduler, OneCycleLR):
            # Step every batch
            self.lr_scheduler.step()
        elif current_batch == 0 and self.lr_scheduler is not None:
            # Step once per epoch
            self.lr_scheduler.step()

        super().step(epoch)

    def get_lr(self):
        return self.lr_scheduler.get_lr()

    def _update_optimizer(self):
        params = self.schedulers[self.active_milestone]
        # Re-initialize optimizer using the default values
        args = copy.deepcopy(self.optimizer.defaults)
        # Override parameters for this milestone
        args.update(params.get("optimizer_args", {}))
        # Update parameters for all parameter groups
        for group in self.optimizer.param_groups:
            group.update(args)

    def _update_lr_scheduler(self):
        params = self.schedulers[self.active_milestone]
        lr_scheduler_class = params.get("lr_scheduler_class", None)
        if lr_scheduler_class is not None:
            lr_scheduler_args = params.get("lr_scheduler_args", None)
            for group in self.optimizer.param_groups:
                # reset initial_lr for new scheduler
                group.pop("initial_lr", None)

            self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_args)
