# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from pprint import pformat

from torch.optim.lr_scheduler import OneCycleLR

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.vernon.expansions.has_lr_scheduler import HasLRScheduler
from nupic.research.frameworks.vernon.experiment_utils import create_lr_scheduler

__all__ = [
    "FixedLRSchedule",
]


class FixedLRSchedule(HasLRScheduler):
    """
    Use an LR schedule that steps every batch or every epoch.
    """
    def setup_experiment(self, config):
        """
        :param config:
            - lr_scheduler_class: Class of lr-scheduler
            - lr_scheduler_args: dict of args to pass to lr-class
            - lr_scheduler_step_every_batch: whether to step every batch
        """
        super().setup_experiment(config)

        assert hasattr(self, "lr_scheduler"), (
            "Must use HasLRScheduler or similar expansion"
        )
        assert hasattr(self, "total_batches"), (
            "Must set self.total_batches"
        )

        self.lr_scheduler = self.create_lr_scheduler(
            config, self.optimizer, self.total_batches)
        if self.lr_scheduler is not None:
            lr_scheduler_class = self.lr_scheduler.__class__.__name__
            lr_scheduler_args = config.get("lr_scheduler_args", {})
            self.logger.info("LR Scheduler class: " + lr_scheduler_class)
            self.logger.info("LR Scheduler args:")
            self.logger.info(pformat(lr_scheduler_args))
            self.logger.info("steps_per_epoch=%s", self.total_batches)

        self.step_lr_every_batch = config.get("lr_scheduler_step_every_batch", False)
        if isinstance(self.lr_scheduler, (OneCycleLR, ComposedLRScheduler)):
            self.step_lr_every_batch = True

    @classmethod
    def create_lr_scheduler(cls, config, optimizer, total_batches):
        """
        Create lr scheduler from an ImagenetExperiment config
        :param config:
            - lr_scheduler_class: (optional) Class of lr-scheduler
            - lr_scheduler_args: (optional) dict of args to pass to lr-class
        :param optimizer: torch optimizer
        :param total_batches: number of batches/steps in an epoch
        """
        lr_scheduler_class = config.get("lr_scheduler_class", None)
        if lr_scheduler_class is not None:
            lr_scheduler_args = config.get("lr_scheduler_args", {})
            return create_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_class=lr_scheduler_class,
                lr_scheduler_args=lr_scheduler_args,
                steps_per_epoch=total_batches)

    def post_batch(self, **kwargs):
        super().post_batch(**kwargs)
        if self.step_lr_every_batch:
            self.lr_scheduler.step()

    def post_epoch(self):
        super().post_epoch()
        if self.lr_scheduler is not None and not self.step_lr_every_batch:
            self.logger.debug(
                "End of epoch %s LR/weight decay before step: %s/%s",
                self.current_epoch, self.get_lr(), self.get_weight_decay())
            self.lr_scheduler.step()
        self.logger.debug(
            "End of epoch %s LR/weight decay after step: %s/%s",
            self.current_epoch, self.get_lr(), self.get_weight_decay())

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        exp = "FixedLRSchedule"

        # Extended methods
        eo["post_batch"].append(exp + ": Maybe step lr_scheduler")
        eo["post_epoch"].append(exp + ": Maybe step lr_scheduler")

        return eo
