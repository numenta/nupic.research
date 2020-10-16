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

import math
from collections import defaultdict

from torchvision import transforms

from nupic.research.frameworks.pytorch.dataset_utils.samplers import (
    TaskDistributedSampler,
    TaskRandomSampler,
)
from nupic.research.frameworks.vernon.evaluation_metrics import ContinualLearningMetrics
from nupic.research.frameworks.vernon.experiments.supervised_experiment import (
    SupervisedExperiment,
)

__all__ = [
    "ContinualLearningExperiment",
]


class ContinualLearningExperiment(ContinualLearningMetrics, SupervisedExperiment):

    def setup_experiment(self, config):

        super().setup_experiment(config)
        # Override epochs to validate to not validate within the inner loop over epochs
        self.epochs_to_validate = []

        self.current_task = 0

        # Defines how many classes should exist per task
        self.num_tasks = config.get("num_tasks", 1)

        self.num_classes = config.get("num_classes", None)
        assert self.num_classes is not None, "num_classes should be defined"

        self.num_classes_per_task = math.floor(self.num_classes / self.num_tasks)

        # Applying target transform depending on type of CL task
        # Task - we know the task, so the network is multihead
        # Class - we don't know the task, network has as many heads as classes
        self.cl_experiment_type = config.get("cl_experiment_type", "class")
        if self.cl_experiment_type == "task":
            self.logger.info("Overriding target transform")
            self.dataset_args["target_transform"] = (
                transforms.Lambda(lambda y: y % self.num_classes_per_task)
            )

        # Whitelist evaluation metrics
        self.evaluation_metrics = config.get(
            "evaluation_metrics", ["eval_all_visited_tasks"]
        )
        for metric in self.evaluation_metrics:
            if not hasattr(self, metric):
                raise ValueError(f"Metric {metric} not available.")

    def should_stop(self):
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """
        return self.current_task >= self.num_tasks

    def run_task(self):
        """Run outer loop over tasks"""
        # configure the sampler to load only samples from current task
        self.logger.info("Training...")
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Run epochs, inner loop
        # TODO: return the results from run_epoch
        for _ in range(self.epochs):
            self.run_epoch()

        ret = {}
        for metric in self.evaluation_metrics:
            eval_function = getattr(self, metric)
            temp_ret = eval_function()
            self.logger.debug(temp_ret)
            for k, v in temp_ret.items():
                ret[f"{metric}__{k}"] = v

        # TODO: Fix aggregate results function to not
        # require these parameters, in order to be more flexible
        ret.update(
            timestep_begin=0,
            timestep_end=1,
            learning_rate=self.get_lr()[0],
            extra_val_results=[],
        )

        self.current_task += 1
        return ret

    @classmethod
    def aggregate_results(cls, results):
        """Run validation in single GPU"""
        return results[0]

    @classmethod
    def create_train_sampler(cls, config, dataset):
        return cls.create_task_sampler(config, dataset, train=True)

    @classmethod
    def create_validation_sampler(cls, config, dataset):
        return cls.create_task_sampler(config, dataset, train=False)

    @classmethod
    def create_task_sampler(cls, config, dataset, train):
        # Assume dataloaders are already created
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(dataset):
            class_indices[target].append(idx)

        # Defines how many classes should exist per task
        num_tasks = config.get("num_tasks", 1)
        num_classes = config.get("num_classes", None)
        assert num_classes is not None, "num_classes should be defined"
        num_classes_per_task = math.floor(num_classes / num_tasks)

        task_indices = defaultdict(list)
        for i in range(num_tasks):
            for j in range(num_classes_per_task):
                task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])

        # Change the sampler in the train loader
        distributed = config.get("distributed", False)
        if distributed and train:
            sampler = TaskDistributedSampler(
                dataset,
                task_indices
            )
        else:
            # TODO: implement a TaskDistributedUnpaddedSampler
            # mplement the aggregate results
            # after above are implemented, remove this if else
            sampler = TaskRandomSampler(task_indices)

        return sampler

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "ContinualLearningExperiment"
        eo["create_train_sampler"] = [exp + ".create_train_sampler"]
        eo["create_validation_sampler"] = [exp + ".create_validation_sampler"]
        eo["create_task_sampler"] = [exp + ".create_task_sampler"]
        eo["run_task"] = [exp + ".run_task"]
        eo["should_stop"] = [exp + ".should_stop"]
        eo["aggregate_results"] = [exp + ".aggregate_results"]
        return eo
