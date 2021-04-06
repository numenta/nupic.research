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


class LRRangeTestMixin:
    """
    Mixin for the LR-range test defined in section 4.1 of "A Disciplined Approach to
    Neural Network Hyper-Parameters"
        - https://arxiv.org/pdf/1803.09820.pdf

    To use this mixin, define min_lr and max_lr through the config and then run a small
    number of steps of training (e.g. 100 steps). Throughout training the lr will be
    gradually increased either linearly or exponentially. Next, review the plotted
    training loss. This mixin won't calculate the max_lr you should use for training
    with a OneCycle schedule. You must inspect the loss yourself and observe when the
    loss begins to increase. This inflection point is where you should set your max_lr.
    Note, sometimes the ideal max_lr turns out an order of magnitude lower; you know
    this is the case if your training loss increases at all during training. For your
    min_lr, the author recommends using 10-20 times lower than the max_lr.

    Params to add to 'mixin_args':
    :param min_lr: starting lr
    :param max_lr: ending lr; presumed to be larger than min_lr
    :param test_mode: either linear or exponential
    """
    def __init__(self, *args, **kwargs):

        # The LambdaLR will multiply this base lr of 1 times the one at the given step.
        kwargs["args"].learning_rate = 1

        # Log so that the lr is recorder every step.
        kwargs["args"].logging_steps = 1

        # Turn off eval since it's satisfactory to just look at the training loss.
        kwargs["args"].do_eval = False

        super().__init__(*args, **kwargs)

        self.min_lr = self.args.mixin_args.get("min_lr", None)
        self.max_lr = self.args.mixin_args.get("max_lr", None)
        self.test_mode = self.args.mixin_args.get("test_mode", "linear")

        assert isinstance(self.min_lr, float)
        assert isinstance(self.max_lr, float)
        assert self.test_mode in ["linear", "exponential"]

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Create a linearly or exponentially increasing lr schedule. This overrides super
        in a way that just customizes the lr scheduler while the optimizer remains the
        default.
        """

        # Set lr scheduler to dummy variable so it's not created in the call to super.
        self.lr_scheduler = ...

        # Create just the optimizer.
        super().create_optimizer_and_scheduler(num_training_steps)

        # Create a lr scheduler that ramps up either linearly or exponentially.
        total_steps = num_training_steps
        min_lr = self.min_lr
        max_lr = self.max_lr

        # Linearly increase lr
        if self.test_mode == "linear":
            def lr_lambda(step: int):
                return (max_lr - min_lr) / (total_steps - 1) * step + min_lr

        # Exponentially increase lr
        elif self.test_mode == "exponential":
            def lr_lambda(step):
                return (max_lr / min_lr) ** (step / (total_steps - 1)) * min_lr

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
