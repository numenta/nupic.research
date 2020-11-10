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

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from nupic.research.frameworks.continual_learning.maml_utils import clone_model
from nupic.research.frameworks.pytorch.dataset_utils.samplers import TaskRandomSampler
from nupic.research.frameworks.pytorch.model_utils import (
    filter_params,
    get_parent_module,
)
from nupic.research.frameworks.vernon.experiments.supervised_experiment import (
    SupervisedExperiment,
)

__all__ = [
    "MetaContinualLearningExperiment",
]


class MetaContinualLearningExperiment(SupervisedExperiment):
    """
    Experiment class for meta-continual learning (based on OML and ANML meta-continual
    learning setups). There are 2 main phases in a meta-continual learning setup:

        - meta-training: Meta-learning representations for continual learning over all
        tasks
        - meta-testing: Montinual learning of new tasks and model evaluation

    where learning a "task" corresponds to learning a single classification label. More
    specifically, each phase is divided into its own training and testing phase, hence
    we have 4 such phases.

        - meta-training training: Train the inner loop learner (i.e., slow parameters)
        for a specific task
        - meta-training testing: Train the outer loop learner (i.e., fast parameters)
        by minimizing the test loss

        - meta-testing training: Train the inner loop learner continually on a sequence
        of holdout tasks
        - meta-testing testing: Evaluate the inner loop learner on the same tasks

    The parameters for a model used in a meta-continual learning setup are broken down
    into 2 groups:

        - fast parameters: Used to update slow parameters via online learning during
        meta-training training, and updated along with slow parameters during
        meta-training testing
        - slow parameters: Updated during the outer loop (meta-training testing)
    """

    def setup_experiment(self, config):
        """
        Configure the experiment for training

        :param config: Dictionary containing the configuration parameters, most of
        which are defined in SupervisedExperiment, but some of which are specific to
        MetaContinualLearningExperiment

            - experiment_class: Class used to run experiments, specify
            `MetaContinualLearningExperiment` for meta-continual learning
            - adaptation_lr: Learning rate used to update the fast parameters during
            the inner loop of the meta-training phase
            - tasks_per_epoch: Number of different classes used for training during the
            execution of the inner loop
            - slow_batch_size: Number of examples in a single batch used to update the
            slow parameters during the meta-training testing phase, where the examples
            are sampled from tasks_per_epoch difference tasks
            - replay_batch_size: Number of examples in a single batch sampled from all
            data, also used to train the slow parameters (the replay batch is used to
            sample examples to update the slow parameters during meta-training testing
            to prevent the learner from forgetting other tasks)
            - num_fast_steps: number of sequential steps to take in the inner loop per
                              every outer loop
            - train_train_sample_size: number of images per class to sample from for
                                      meta-training; same size will be used in both
                                      the inner and outer loops. The rest of the
                                      images will be used for a validation step.
            - fast_params: list of regex patterns identifying which params to
                           update during meta-train training
        """
        if "num_classes" not in config["model_args"]:
            # manually set `num_classes` in `model_args`
            num_classes = config["num_classes"]
            config["model_args"]["num_classes"] = num_classes

        super().setup_experiment(config)

        self.epochs_to_validate = []
        self.tasks_per_epoch = config.get("tasks_per_epoch", 1)
        self.num_classes = min(
            config.get("num_classes", 50),
            self.train_fast_loader.sampler.num_classes
        )

        self.adaptation_lr = config.get("adaptation_lr", 0.03)

        self.batch_size = config.get("batch_size", 5)
        self.val_batch_size = config.get("val_batch_size", 15)
        self.slow_batch_size = config.get("slow_batch_size", 64)
        self.replay_batch_size = config.get("replay_batch_size", 64)
        self.num_fast_steps = config.get("num_fast_steps", 1)

        assert "fast_params" in config
        fast_named_params = filter_params(self.model,
                                          include_patterns=config["fast_params"])
        self.fast_param_names = list(fast_named_params.keys())
        self.logger.info(f"Setup: fast_param_names={self.fast_param_names}")

        if self.num_fast_steps > len(self.train_fast_loader):
            self.logger.warning(
                "The num_fast_steps given is greater than the len of images available "
                " for the inner loop. This should ideally be no more than:\n"
                " train_train_sample_size * tasks_per_epoch / batch_size"
            )

    def create_loaders(self, config):
        """Create train and val dataloaders."""

        main_set = self.load_dataset(config, train=True)

        # All loaders share tasks and dataset, but different indices and batch sizes.
        self.train_fast_loader = self.create_train_dataloader(config, main_set)
        self.train_slow_loader = self.create_slow_train_dataloader(config, main_set)
        self.train_replay_loader = self.create_replay_dataloader(config, main_set)
        self.val_fast_loader = self.create_validation_dataloader(config, main_set)

        # For pre/post epoch and batch processing slow loader is equiv to train loader
        self.train_loader = self.train_slow_loader

    @classmethod
    def create_train_sampler(cls, config, dataset):
        """Sampler for meta-train training."""
        sample_size = config.get("train_train_sample_size", 5)
        class_indices = cls.compute_class_indices(config, dataset,
                                                  mode="train",
                                                  sample_size=sample_size)
        return cls.create_sampler(config, dataset, class_indices)

    @classmethod
    def create_train_slow_sampler(cls, config, dataset):
        """Sampler for meta-train testing. Uses same images as for train-training."""
        sample_size = config.get("train_train_sample_size", 5)
        class_indices = cls.compute_class_indices(config, dataset,
                                                  mode="train",
                                                  sample_size=sample_size)
        return cls.create_sampler(config, dataset, class_indices)

    @classmethod
    def create_replay_sampler(cls, config, dataset):
        """Sampler used to augment meta-train testing; "replays" previous classes."""
        class_indices = cls.compute_class_indices(config, dataset, mode="all")
        return cls.create_sampler(config, dataset, class_indices)

    @classmethod
    def create_validation_sampler(cls, config, dataset):
        """Sampler used to validate meta-training phase."""
        sample_size = config.get("train_train_sample_size", 5)
        class_indices = cls.compute_class_indices(config, dataset,
                                                  mode="test",
                                                  sample_size=sample_size)
        return cls.create_sampler(config, dataset, class_indices)

    @classmethod
    def compute_class_indices(cls, config, dataset, mode="all", sample_size=None):
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(dataset):
            class_indices[target].append(idx)

        if mode == "train":
            assert isinstance(sample_size, int)
            for c in class_indices:
                class_indices[c] = class_indices[c][:sample_size]
        elif mode == "test":
            assert isinstance(sample_size, int)
            for c in class_indices:
                class_indices[c] = class_indices[c][sample_size:]
        elif mode == "all":
            pass
        else:
            raise ValueError(f"Received unexpected mode: {mode}")

        return class_indices

    @classmethod
    def create_sampler(cls, config, dataset, class_indices):
        """
        Provides a hook for a distributed experiment.
        """
        return TaskRandomSampler(class_indices)

    @classmethod
    def create_slow_train_dataloader(cls, config, dataset):
        sampler = cls.create_train_slow_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("slow_batch_size", 64),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def create_replay_dataloader(cls, config, dataset):
        sampler = cls.create_replay_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("replay_batch_size", 64),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    def run_epoch(self):

        self.pre_epoch()

        self.optimizer.zero_grad()

        # Sample tasks for inner loop.
        tasks_train = np.random.choice(
            self.num_classes, self.tasks_per_epoch, replace=False
        )

        # Run pre_task; For instance, may reset parameters as needed.
        self.pre_task(tasks_train)

        # Clone model - clone fast params and the slow params. The latter will be frozen
        cloned_adaptation_net = self.clone_model()

        # Inner loop: Train over sampled tasks.
        for task in tasks_train:
            self.run_task(task, cloned_adaptation_net)

        # Sample from the replay set.
        self.train_replay_loader.sampler.set_active_tasks(list(range(self.num_classes)))
        replay_data, replay_target = next(iter(self.train_replay_loader))

        # Sample from the slow set.
        slow_data, slow_target = [], []
        for task in tasks_train:
            self.train_slow_loader.sampler.set_active_tasks(task)
            x, y = next(iter(self.train_slow_loader))
            slow_data.append(x)
            slow_target.append(y)

        # Concatenate the slow and replay set.
        slow_data = torch.cat(slow_data + [replay_data]).to(self.device)
        slow_target = torch.cat(slow_target + [replay_target]).to(self.device)

        # Take step for outer loop. This will backprop through to the original
        # slow and fast params.
        output = cloned_adaptation_net(slow_data)
        loss = self._loss_function(output, slow_target)
        loss.backward()

        self.optimizer.step()

        # Report statistics for the outer loop
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(slow_target.view_as(pred)).sum().item()
        total = output.shape[0]
        results = {
            "total_correct": correct,
            "total_tested": total,
            "mean_loss": loss.item(),
            "mean_accuracy": correct / total if total > 0 else 0,
            "learning_rate": self.get_lr()[0],
        }
        self.logger.debug(results)

        self.post_epoch()

        self.current_epoch += 1

        return results

    def pre_task(self, tasks):
        """
        Run any necessary pre-task logic for the upcoming tasks.

        :param tasks: list of task about to be ran.
        """
        pass

    def run_task(self, task, cloned_adaptation_net):
        self.train_fast_loader.sampler.set_active_tasks(task)

        # Meta-train training. Use no more than `num_fast_steps` sequential updates.
        for i, (data, target) in enumerate(self.train_fast_loader):
            if i >= self.num_fast_steps:
                break

            data = data.to(self.device)
            target = target.to(self.device)
            train_loss = self._loss_function(
                cloned_adaptation_net(data), target
            )
            # Update in place
            self.adapt(cloned_adaptation_net, train_loss)

        # See if there are images to validate on. If 'train_train_sample_size'
        # is equivalent to the number of images per class, then there won't be any.
        if len(self.val_fast_loader) == 0:
            return

        # Run and log validation for given task.
        with torch.no_grad():
            self.val_fast_loader.sampler.set_active_tasks(task)

            data, target = next(iter(self.val_fast_loader))
            data = data.to(self.device)
            target = target.to(self.device)

            preds = cloned_adaptation_net(data)
            valid_error = self._loss_function(preds, target)
            valid_error /= len(data)
            self.logger.debug(f"Valid error meta train training: {valid_error}")

            # calculate accuracy
            preds = preds.argmax(dim=1).view(target.shape)
            valid_accuracy = (preds == target).sum().float() / target.size(0)
            self.logger.debug(f"Valid accuracy meta train training: {valid_accuracy}")

    @classmethod
    def update_params(cls, named_params, model, loss, lr):
        """
        Takes a gradient step on the loss and updates the cloned parameters in place.
        """
        named_params = dict(named_params)
        params = list(named_params.values())
        gradients = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=True
        )

        if gradients is not None:
            for g, (name, p) in zip(gradients, named_params.items()):
                if g is not None:
                    updated = p.add(g, alpha=-lr)

                    # Update in-place in a way that preserves grads.
                    parent_module = get_parent_module(model, name)
                    base_name = name.split(".")[-1]
                    parent_module._parameters[base_name] = updated

    def adapt(self, cloned_adaptation_net, train_loss):
        named_fast_params = self.get_named_fast_params(cloned_adaptation_net)
        self.update_params(
            named_fast_params, cloned_adaptation_net, train_loss,
            self.adaptation_lr
        )

    def clone_model(self, keep_as_reference=None):
        """
        Clones self.model by cloning some of the params and keeping those listed
        specified `keep_as_reference` via reference.
        """
        return clone_model(self.model, keep_as_reference=None)

    def get_named_fast_params(self, clone=None):
        """Filter out the params from fast_param_names."""
        model = self.get_model(clone=clone)
        named_fast_params = {}
        for n, p in model.named_parameters():
            if n in self.fast_param_names:
                named_fast_params[n] = p

        return named_fast_params

    def get_model(self, clone=None):
        model = clone if clone is not None else self.model
        return model

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "MetaContinualLearningExperiment"
        eo["create_loaders"] = [exp + ".create_loaders"]
        eo["create_fast_slow_loaders"] = [exp + ".create_fast_slow_loaders"]
        eo["create_sampler"] = [exp + ".create_sampler"]
        eo["create_train_sampler"] = [exp + ".create_train_sampler"]
        eo["create_validation_sampler"] = [exp + ".create_validation_sampler"]
        eo["create_replay_sampler"] = [exp + ".create_replay_sampler"]
        eo["create_task_sampler"] = [exp + ".create_task_sampler"]
        eo["create_slow_train_dataloader"] = [exp + ".create_slow_train_dataloader"]
        eo["run_epoch"] = [exp + ".run_epoch"]
        eo["pre_task"] = [exp + ".pre_task"]
        eo["run_task"] = [exp + ".run_task"]
        eo["update_params"] = [exp + ".update_params"]
        eo["adapt"] = [exp + ".adapt"]
        eo["clone_model"] = [exp + ".clone_model"]
        eo["get_named_fast_params"] = [exp + ".get_named_fast_params"]
        eo["get_model"] = [exp + ".get_model"]
        return eo
