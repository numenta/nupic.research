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

from functools import partial

from torch.optim.lr_scheduler import LambdaLR
from transformers import IntervalStrategy


class LRRangeTestMixin:
    """
    Mixin for the LR-range test defined in section 4.1 of "A Disciplined Approach to
    Neural Network Hyper-Parameters"
        - https://arxiv.org/pdf/1803.09820.pdf

    This test helps decide what to set your max_lr for a OneCycle LR schedule.

    To use this mixin, define min_lr and max_lr through the config. This is distinct
    from the range you'll use for the OneCycle schedule and should be very wide.
    Ideally, it should be a super-set of the range you're looking for, something like
    like 1e-5 to 1e-1. Then start a small number of steps of training (e.g. 100 steps)
    during which the lr will be gradually increased through the specified range. After
    training, visually inspect the plotted training and evaluations loss. Identify the
    point where the validation loss begins to increase, but where the training loss is
    still decreasing. This inflection point marks the lr at which training becomes
    unstable. Thus, this should be you max_lr for the OneCycle schedule. For your
    min_lr, the author recommends using 10-20 times lower than the max_lr.

    Params to add to 'trainer_mixin_args':
    :param min_lr: starting lr
    :param max_lr: ending lr; presumed to be larger than min_lr
    :param test_mode: either increase the lr linearly or exponentially; in practice, the
                      linear mode has been easier to interpret so this is the default
    :param eval_dataset_fraction: percentage of the dataset to use during evaluation;
                                  the same subset is used for evaluation after every
                                  step of training; defaults to 5%
    """
    def __init__(self, *args, **kwargs):

        # The LambdaLR will multiply this base lr of 1 times the one at the given step.
        kwargs["args"].learning_rate = 1

        # Log so that the lr is recorder every step.
        kwargs["args"].logging_steps = 1

        # Turn on evaluation after every step of training.
        kwargs["args"].evaluation_strategy = IntervalStrategy("steps")
        kwargs["args"].eval_steps = 1

        super().__init__(*args, **kwargs)

        mixin_args = self.args.trainer_mixin_args
        self.min_lr = mixin_args.get("min_lr", None)
        self.max_lr = mixin_args.get("max_lr", None)
        self.test_mode = mixin_args.get("test_mode", "linear")

        # Use only a fraction of the dataset for evaluation.
        eval_dataset_fraction = mixin_args.get("eval_dataset_fraction", 0.05)
        self.eval_dataset = self.eval_dataset.shard(
            index=1, num_shards=int(1 / eval_dataset_fraction)
        )

        assert isinstance(self.min_lr, float)
        assert isinstance(self.max_lr, float)
        assert self.test_mode in ["linear", "exponential"]

    def create_scheduler(self, num_training_steps: int):
        """
        Create a lr scheduler that ramps up either linearly or exponentially.
        """

        total_steps = num_training_steps
        min_lr = self.min_lr
        max_lr = self.max_lr

        lr_lambda = partial(
            linear_lr if self.test_mode == "linear" else exponential_lr,
            total_steps=total_steps,
            min_lr=min_lr,
            max_lr=max_lr
        )

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)


def linear_lr(step, total_steps, min_lr, max_lr):
    return (max_lr - min_lr) / (total_steps - 1) * step + min_lr


def exponential_lr(step, total_steps, min_lr, max_lr):
    return (max_lr / min_lr) ** (step / (total_steps - 1)) * min_lr
