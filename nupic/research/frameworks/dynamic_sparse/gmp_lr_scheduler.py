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

from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "ThreeStageGMPLR",
]


class ThreeStageGMPLR(LambdaLR):
    """
    This is an lr schedule for Gradual Magnitude Pruning (GMP) consisting of three
    stages:
        1) warmup - linearly ramp up from 0 to max_lr; this differs from the paper
                    which uses a constant lr
        2) pruning - maintain a constant max_lr
        3) cooldown - decay the learning rate twice (like a StepLR)

    :param max_lr: this is the maximum lr reached
    :param min_lr: initial lr during the warmup phase
    :param warmup_steps: number of steps for the warmup phase
    :param cooldown_steps: number of steps for the cooldown phase
    :param total_steps: total number of steps; used to derive pruning steps
    :param cooldown_gamma: how much to decay the lr during cooldown
                           e.g. lr <- lr * cooldown_gamma; used to decay the lr twice
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_steps,
        cooldown_steps,
        total_steps,
        cooldown_gamma,
        min_lr=0,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.total_steps = total_steps
        self.pruning_steps = total_steps - warmup_steps - cooldown_steps
        self.cooldown_gamma = 0.1
        assert int(cooldown_steps / 3) >= 2, (
            "Too few `cooldown_steps`. The lr will be decayed twice throughout the "
            "cooldown period."
        )

        super().__init__(optimizer, self.composed_lr)

    def composed_lr(self, step: int):
        """
        LR schedule for GMP Pruning consisting of three phases. Here `step` starts at 0.
        """

        # Step starts at 0, but will make it start at 1.
        train_step = step + 1

        # Phase 1: Warm-up phase - linear warmup.
        if train_step < self.warmup_steps:
            return self.warmup_lr(step)

        # Phase 3: Cool-down phase - StepLR like decay
        elif train_step > self.total_steps - self.cooldown_steps:
            return self.cooldown_lr(step)

        # Phase 2: Pruning phase - constant lr.
        else:
            return self.pruning_lr(step)

    # Warm-up phase.
    def warmup_lr(self, step: int):
        """Linearly ramp up the lr from min_lr to max_lr in `warmup_steps`"""
        slope = (self.max_lr - self.min_lr) / (self.warmup_steps - 1)
        return slope * step + self.min_lr

    # Pruning phase
    def pruning_lr(self, step: int):
        """Keep the lr constant"""
        return self.max_lr

    # Cool-down phase
    def cooldown_lr(self, step: int):
        """
        Similar to StepLR, decay the learning twice throughout the cooldown phase.
        Right now it only supports decaying the lr twice.
        """

        step_size = int(self.cooldown_steps / 3)
        step_into_cooldown = (step + 1) - self.warmup_steps - self.pruning_steps

        # Decay twice when past `2 * step_size`
        if step_into_cooldown > 2 * step_size:
            return self.max_lr * self.cooldown_gamma ** 2

        # Decay one when past `step_size`
        elif step_into_cooldown > step_size:
            return self.max_lr * self.cooldown_gamma

        # Don't decay when not past the step_size.
        else:
            return self.max_lr
