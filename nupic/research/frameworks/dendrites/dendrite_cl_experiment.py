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

from nupic.research.frameworks.dendrites import (
    evaluate_dendrite_model,
    train_dendrite_model,
)
from nupic.research.frameworks.vernon import ContinualLearningExperiment

__all__ = [
    "DendriteContinualLearningExperiment",
]


class DendriteContinualLearningExperiment(ContinualLearningExperiment):

    def setup_experiment(self, config):
        super().setup_experiment(config)

    def run_task(self):
        """
        Run the current task.
        """
        # configure the sampler to load only samples from current task
        self.logger.info("Training task %d...", self.current_task)
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Run epochs, inner loop
        # TODO: return the results from run_epoch
        self.current_epoch = 0
        for _ in range(self.epochs):
            self.run_epoch()

        # TODO: put back evaluation_metrics from cl_experiment
        # TODO: add option to run validate on all tasks at end of training.
        # TODO: add option to run validate on one task at a time during training
        ret = {}
        if self.current_task in self.tasks_to_validate:
            self.val_loader.sampler.set_active_tasks(range(self.num_tasks))
            ret = self.validate()
            self.val_loader.sampler.set_active_tasks(self.current_task)

        ret.update(
            learning_rate=self.get_lr()[0],
        )

        self.current_task += 1

        if self.reset_optimizer_after_task:
            self.optimizer = self.recreate_optimizer(self.model)

        print("run_task ret: ", ret)
        return ret

    def train_epoch(self):
        # TODO: take out constants in the call below. How do we determine num_labels?
        train_dendrite_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            share_labels=True,
            num_labels=10,
            post_batch_callback=self.post_batch_wrapper
        )

    def validate(self, loader=None):
        """
        Run validation on the currently active tasks.
        """
        if loader is None:
            loader = self.val_loader

        # TODO: take out constants in the call below
        return evaluate_dendrite_model(model=self.model,
                                       loader=loader,
                                       device=self.device,
                                       criterion=self.error_loss,
                                       share_labels=True, num_labels=10)
