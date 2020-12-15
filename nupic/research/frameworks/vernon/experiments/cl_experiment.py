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
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms

from nupic.research.frameworks.pytorch.dataset_utils.samplers import TaskRandomSampler
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
        self.cl_experiment_type = config.get("cl_experiment_type", "class")
        if self.cl_experiment_type == "task":
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
        self.train_model = config.get("train_model_func", self.train_cl_model)
        self.evaluate_model = config.get("evaluate_model_func", self.evaluate_cl_model)

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

    def get_active_classes(self, task_num):
        """
        Returns a list of label indices that are "active" during training under the
        continual learning scenario specified by the config. In the "task" scenario,
        only the classes that are being trained are active. In the "class" scenario,
        all tasks that the model has previously observed are active. More information
        about active classes can be found here: https://arxiv.org/abs/1904.07734

        :param task_num: zero-based index of the task
        """
        if self.cl_experiment_type == "task":
            return [label for label in range(
                self.num_classes_per_task * task_num,
                self.num_classes_per_task * (task_num + 1)
            )]
        elif self.cl_experiment_type == "class":
            return [label for label in range(
                self.num_classes_per_task * (task_num + 1)
            )]
        elif self.cl_experiment_type == "domain":
            raise NotImplementedError

    def train_cl_model(
        self,
        model,
        loader,
        optimizer,
        device,
        freeze_params=None,
        criterion=F.nll_loss,
        complexity_loss_fn=None,
        batches_in_epoch=sys.maxsize,
        pre_batch_callback=None,
        post_batch_callback=None,
        progress_bar=None,
        transform_to_device_fn=None
    ):
        """
        Trains a givn model over a single epoch similar to `train_model` used by
        `SupervisedExperiment`, but this function is compatible with the "task" and
        "class" continual learning scenarios.

        Note that the parameters `complexity_loss_fn`, `progress_bar`, and
        `transform_to_device_fn` are unused, yet still included in the function
        signature to be consistent with that of `train_model`.
        """
        model.train()
        async_gpu = loader.pin_memory
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= batches_in_epoch:
                break

            num_images = len(target)
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)
            t1 = time.time()

            if pre_batch_callback is not None:
                pre_batch_callback(model=model, batch_idx=batch_idx)

            optimizer.zero_grad()
            # Compute loss only for "active" output units, and only backpropogate errors
            # for these units when calling loss.backward()
            active_classes = self.get_active_classes(self.current_task)
            output = model(data)
            output = output[:, active_classes]
            error_loss = criterion(output, target)

            del data, target, output

            t2 = time.time()
            error_loss.backward()

            t3 = time.time()
            optimizer.step()
            t4 = time.time()

            if post_batch_callback is not None:
                time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                               + "weight update: {:.3f}s").format(t1 - t0, t2 - t1,
                                                                  t3 - t2, t4 - t3)
                post_batch_callback(model=model,
                                    error_loss=error_loss.detach(),
                                    complexity_loss=None,
                                    batch_idx=batch_idx,
                                    num_images=num_images,
                                    time_string=time_string)
            del error_loss
            t0 = time.time()

    def evaluate_cl_model(
        self,
        model,
        loader,
        device,
        batches_in_epoch=sys.maxsize,
        criterion=F.nll_loss,
        complexity_loss_fn=None,
        progress=None,
        post_batch_callback=None,
        transform_to_device_fn=None,
    ):
        """
        Evaluates a given model similar to `evaluate_model` used by
        `SupervisedExperiment`, but this function is compatible with the "task" and
        "class" continual learning scenarios.

        Note that the parameters `complexity_loss_fn`, `progress`, and
        `transform_to_device_fn` are unused, yet still included in the function
        signature to be consistent with that of `evaluate_model`.
        """
        model.eval()
        total = 0

        # Perform accumulation on device, avoid paying performance cost of .item()
        loss = torch.tensor(0., device=device)
        correct = torch.tensor(0, device=device)

        async_gpu = loader.pin_memory

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= batches_in_epoch:
                    break

                data = data.to(device, non_blocking=async_gpu)
                target = target.to(device, non_blocking=async_gpu)

                # Compute loss only for "active" output units, and only backpropogate
                # errors for these units when calling loss.backward()
                active_classes = self.get_active_classes(self.current_task)
                output = model(data)
                output = output[:, active_classes]

                loss += criterion(output, target, reduction="sum")
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum()
                total += len(data)

                if post_batch_callback is not None:
                    post_batch_callback(batch_idx=batch_idx, target=target,
                                        output=output, pred=pred)

        correct = correct.item()
        loss = loss.item()

        result = {
            "total_correct": correct,
            "total_tested": total,
            "mean_loss": loss / total if total > 0 else 0,
            "mean_accuracy": correct / total if total > 0 else 0,
        }

        return result

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
