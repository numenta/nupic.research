# ------------------------------------------------------------------------------
# Copyright (C) 2020, Numenta, Inc. All rights reserved.
#
# The information and source code contained herein is the
# exclusive property of Numenta Inc.  No part of this software
# may be used, reproduced, stored or distributed in any form,
# without explicit written authorization from Numenta Inc.#
# ------------------------------------------------------------------------------

from copy import deepcopy

from nupic.research.frameworks.pytorch.imagenet import mixins
from nupic.research.frameworks.pytorch.lr_scheduler import LinearLRScheduler


class LRRangeTest(object):
    """
    Mixin for the LR-range test defined in section 4.1 of "A Disciplined Approach to
    Neural Network Hyper-Parameters"
        - https://arxiv.org/pdf/1803.09820.pdf

    On usage: Make sure that the `required_mixins` are included in your experiment_class
              See `create_lr_test_experiment` to automate this.

    Herein, a min_lr and max_lr are set, and training proceeds for a small number of
    epochs (1-3) while the learning rate is linearly increased. Generally, the point
    at which the training loss begins to curve upwards and increases while the
    validation/test loss still decreases, is considered to be a reasonable choice
    for your max_lr in a cyclical lr-schedule. The same author recommends using 10-20
    times lower this amount for your min_lr.
    """

    # List the required mixins that should be accompanied with this class.
    required_mixins = [mixins.LogEveryLoss, mixins.LogEveryLearningRate]

    def setup_experiment(self, config):
        """
        :param config:
            - epochs: number of epochs in training (recommended 1-3)
            - epochs_to_validate: will be overridden to include all epochs
            - lr_scheduler_class: automatically overridden to LinearLRScheduler
            - lr_scheduler_args: args for the linear-schedule
                - min_lr: starting learning rate
                - max_lr: ending learning rate
        """
        super().setup_experiment(config)

        # Ensure all epochs get validated.
        assert "epochs" in config
        config["epochs_to_validate"] = range(-1, config["epochs"])

        # Save config for later - used to aggregate results.
        self._config = deepcopy(config)

    @classmethod
    def create_lr_scheduler(cls, config, optimizer, total_batches):

        assert "lr_scheduler_args" in config

        lr_scheduler_class = LinearLRScheduler
        lr_scheduler_args = config["lr_scheduler_args"]

        lr_scheduler_args.update(epochs=config["epochs"])
        lr_scheduler_args.update(steps_per_epoch=total_batches)

        return lr_scheduler_class(optimizer, **lr_scheduler_args)

    def post_batch(self, *args, **kwargs):
        """Increase lr after every batch."""
        super().post_batch(*args, **kwargs)
        self.lr_scheduler.step()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].insert(0, cls.__name__ + ": initialize")
        eo["create_lr_scheduler"] = cls.__name__ + ": create_lr_scheduler"
        eo["post_batch"].insert(0, cls.__name__ + ": linearly increase lr")
        return eo


def create_lr_test_experiment(experiment_class):
    """
    This is a helper function to LRRangeTest to ensure your experiment
    class includes the required additional mixins. Specifically, you'll
    want to be sure the lr and training loss are logged after every batch.
    """
    required_mixins = LRRangeTest.required_mixins + [LRRangeTest]
    remaining_mixins = [
        mixin
        for mixin in required_mixins
        if not issubclass(experiment_class, mixin)
    ]

    class Cls(*remaining_mixins, experiment_class):
        pass

    Cls.__name__ = f"{LRRangeTest.__name__}{experiment_class.__name__}"
    return Cls
