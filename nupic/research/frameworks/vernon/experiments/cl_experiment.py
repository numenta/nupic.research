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

from nupic.research.frameworks.pytorch.dataset_utils.samplers import TaskRandomSampler
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.research.frameworks.vernon.experiments.components.evaluation_metrics import (
    ContinualLearningMetrics,
)
from nupic.research.frameworks.vernon.experiments.supervised_experiment import (
    SupervisedExperiment,
)

__all__ = [
    "ContinualLearningExperiment",
]


class ContinualLearningExperiment(
    ContinualLearningMetrics,
    SupervisedExperiment,
):

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
        self.scenario = config.get("scenario", "class")
        if self.scenario == "task":
            self.logger.info("Overriding target transform")
            target_transform = transforms.Lambda(
                lambda y: y % self.num_classes_per_task
            )

            self.train_loader.dataset.targets = target_transform(
                self.train_loader.dataset.targets
            )
            self.val_loader.dataset.targets = target_transform(
                self.val_loader.dataset.targets
            )

        # Set train and validate methods.
        self.train_model = config.get("train_model_func", train_model)
        self.evaluate_model = config.get("evaluate_model_func", evaluate_model)

        # Whitelist evaluation metrics
        self.evaluation_metrics = config.get(
            "evaluation_metrics", ["eval_all_visited_tasks"]
        )
        for metric in self.evaluation_metrics:
            if not hasattr(self, metric):
                raise ValueError(f"Metric {metric} not available.")

    def run_iteration(self):
        return self.run_task()

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

        ret.update(
            learning_rate=self.get_lr()[0],
        )

        self.current_task += 1
        return ret

    @classmethod
    def create_train_sampler(cls, config, dataset):
        task_indices = cls.compute_task_indices(config, dataset)
        return TaskRandomSampler(task_indices)

    @classmethod
    def create_validation_sampler(cls, config, dataset):
        task_indices = cls.compute_task_indices(config, dataset)
        return TaskRandomSampler(task_indices)

    @classmethod
    def compute_task_indices(cls, config, dataset):
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
        return task_indices

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        active_classes = self.get_active_classes()
        return self.evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            active_classes=active_classes,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def train_epoch(self):
        active_classes = self.get_active_classes()
        self.train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch,
            active_classes=active_classes,
            pre_batch_callback=self.pre_batch,
            post_batch_callback=self.post_batch_wrapper,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def get_active_classes(self):
        """
        Returns a list of label indices that are "active" during training in the
        continual learning scenario specified by the config. In the "task" scenario,
        only the classes that are being trained are active. In the "class" scenario,
        all tasks that the model has previously observed are active. More information
        about active classes can be found here: https://arxiv.org/abs/1904.07734
        """
        if self.scenario == "task":
            return [label for label in range(
                self.num_classes_per_task * self.current_task,
                self.num_classes_per_task * (self.current_task + 1)
            )]

        elif self.scenario == "class":
            return [label for label in range(
                self.num_classes_per_task * (self.current_task + 1)
            )]

        else:
            raise Exception("`scenario` must be either 'task' or 'class'")

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "ContinualLearningExperiment"

        # Extended methods
        eo["setup_experiment"].append(exp + ".setup_experiment")

        eo.update(
            # Overwritten methods
            should_stop=[exp + ".should_stop"],
            run_iteration=[exp + ": Call run_task"],
            create_train_sampler=[exp + ".create_train_sampler"],
            create_validation_sampler=[exp + ".create_validation_sampler"],
            aggregate_results=[exp + ".aggregate_results"],
            aggregate_pre_experiment_results=[
                exp + ".aggregate_pre_experiment_results"
            ],

            # New methods
            run_task=[exp + ".run_task"],
        )

        return eo
