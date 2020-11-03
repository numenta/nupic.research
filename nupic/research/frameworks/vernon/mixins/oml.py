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

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.nn.init import kaiming_normal_, zeros_
from torch.optim import Adam
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.model_utils import (
    evaluate_model,
    filter_params,
    train_model,
)
from nupic.research.frameworks.pytorch.models import OMLNetwork


class OnlineMetaLearning(object):
    """
    Implements methods specific to OML, according to original implementation
    Reference: https://github.com/khurramjaved96/mrcl
    """
    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - run_meta_test: whether or not to run the meta-testing phase
            - reset_output_params: whether to reset (i.e. re-init) the output layer
                                   params prior to meta-test training
            - lr_sweep_range: list of learning rates to attempt meta-test training.
                              The best one, according to the meta-test test set,
                              will be chosen and used for the meta-testing phase.
            - num_lr_search_runs: number of runs to attempt an lr-sweep. The one that
                                  achieves the highest test-test accuracy the most
                                  times, i.e. the mode, will be chosen for the
                                  meta-testing phase.
            - num_meta_testing_runs: number of meta-testing phases to run
            - test_train_sample_size: number of images per class to sample from for
                                     meta-testing training. The rest of the
                                     images will be used for meta-test testing.
            - test_train_params: list of regex patterns identifying which params to
                                 update during meta-test training
            - output_layer_params: list of names for the output layer; if this is given
                                   and `reset_output_params=True` these will be reset
                                   prior to meta-test training
        """
        super().setup_experiment(config)
        self.run_meta_test = config.get("run_meta_test", False)
        self.reset_output_params = config.get("reset_output_params", True)
        self.lr_sweep_range = config.get("lr_sweep_range", [1e-1, 1e-2, 1e-3, 1e-4])
        self.num_lr_search_runs = config.get("num_lr_search_runs", 5)
        self.num_meta_testing_runs = config.get("num_meta_testing_runs", 15)

        # Resolve the names of the meta-test training params.
        assert "test_train_params" in config
        test_train_named_params = filter_params(
            self.model,
            include_patterns=config["test_train_params"]
        )
        self.test_train_param_names = list(test_train_named_params.keys())

        # Resolve the names of the output layer params.
        if self.reset_output_params:
            assert "output_layer_params" in config
        output_named_params = filter_params(
            self.model,
            include_names=config.get("output_layer_params", [])
        )
        self.output_param_names = list(output_named_params.keys())

    def create_loaders(self, config):
        super().create_loaders(config)

        eval_set = self.load_dataset(config, train=False)

        self.test_train_loader = self.create_test_train_dataloader(config, eval_set)
        self.test_test_loader = self.create_test_test_dataloader(config, eval_set)

        self.num_classes_eval = min(
            config.get("num_classes_eval", 50),
            self.test_train_loader.sampler.num_classes
        )

    @classmethod
    def create_test_train_sampler(cls, config, dataset):
        """Sampler for meta-test training."""
        sample_size = config.get("test_train_sample_size", 15)
        class_indices = cls.compute_class_indices(config, dataset,
                                                  mode="train",
                                                  sample_size=sample_size)
        return cls.create_sampler(config, dataset, class_indices)

    @classmethod
    def create_test_test_sampler(cls, config, dataset):
        """Sampler for meta-test testing."""
        sample_size = config.get("test_train_sample_size", 15)
        class_indices = cls.compute_class_indices(config, dataset,
                                                  mode="test",
                                                  sample_size=sample_size)
        return cls.create_sampler(config, dataset, class_indices)

    @classmethod
    def create_test_train_dataloader(cls, config, dataset):
        sampler = cls.create_test_train_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("test_train_batch_size", 1),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def create_test_test_dataloader(cls, config, dataset):
        sampler = cls.create_test_test_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("test_test_batch_size", 1),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    def post_epoch(self):
        super().post_epoch()
        if self.should_stop() and self.run_meta_test:
            # meta-training phase complete, perform meta-testing phase
            for num_classes_learned in [10, 50, 100, 200, 600]:
                if num_classes_learned > self.num_classes_eval:
                    break

                # TODO log results in addition to simply printing them to stdout
                results = self.run_meta_testing_phase(num_classes_learned)
                test_train_accs, test_test_accs = results
                print("Accuracy for meta-testing phase over"
                      f" {num_classes_learned} num classes.")
                print("Meta-test training accuracies:", test_train_accs)
                print("Meta-test tests accuracies:", test_test_accs)

    def find_best_lr(self, num_classes_learned):
        """
        This is a simple hyper-parameter search for a good lr:
            1) Sample num_classes_learned classes
            2) Train over the sampled classes; once for each lr
            3) Evaluate the model on a held-out set
            4) Repeat as many times as desired and pick the lr that performs the best
               the most number times
        """

        lr_all = []

        # Grid search over lr
        for _lr_search_runs in range(0, self.num_lr_search_runs):

            # Choose num_classes_learned random classes to train and then test on.
            new_tasks = np.random.choice(
                self.num_classes_eval, num_classes_learned, replace=False
            )

            max_acc = -1000
            for lr in self.lr_sweep_range:

                # Reset output layer weights.
                if self.reset_output_params:
                    output_params = self.get_named_output_params()
                    self.reset_params(output_params.values())

                # Meta-test training.
                test_train_param = self.get_named_test_train_params()
                optim = Adam(test_train_param.values(), lr=lr)
                for task in new_tasks:
                    self.test_train_loader.sampler.set_active_tasks(task)
                    train_model(
                        model=self.model.module,
                        loader=self.test_train_loader,
                        optimizer=optim,
                        device=self.device,
                        criterion=self._loss_function,
                    )

                # Meta-test testing.
                self.test_test_loader.sampler.set_active_tasks(new_tasks)
                results = evaluate_model(
                    model=self.model.module,
                    loader=self.test_test_loader,
                    device=self.device,
                    criterion=self._loss_function,
                )
                correct = results["total_correct"]

                acc = correct / len(self.test_test_loader.sampler.indices)
                if (acc > max_acc):
                    max_acc = acc
                    max_lr = lr

            lr_all.append(max_lr)

        best_lr = float(stats.mode(lr_all)[0][0])
        return best_lr

    def run_meta_testing_phase(self, num_classes_learned):
        """
        Run the meta-testing phase: train over num_classes_learned and then test over a
        held-out set comprised of those same classes (aka the meta-test test set). This
        shows the model's ability to conduct continual learning in a way that allows
        generalization. As well, at the end of this phase, this function also evaluates
        the models performance on the meta-test training set to evaluate it's ability to
        memorize without forgetting.
        """

        lr = self.find_best_lr(num_classes_learned)
        print(f"Found best lr={lr} for num_classes_learned={num_classes_learned}")

        meta_test_test_accuracies = []
        meta_test_train_accuracies = []
        for _current_run in range(0, self.num_meta_testing_runs):

            # Choose num_classes_learned random classes to train and then test on.
            new_tasks = np.random.choice(
                self.num_classes_eval, num_classes_learned, replace=False
            )

            # Reset output layer weights.
            if self.reset_output_params:
                output_params = self.get_named_output_params()
                self.reset_params(output_params.values())

            # Meta-testing training.
            test_train_param = self.get_named_test_train_params()
            optim = Adam(test_train_param.values(), lr=lr)
            for task in new_tasks:
                self.test_train_loader.sampler.set_active_tasks(task)
                train_model(
                    model=self.model.module,
                    loader=self.test_train_loader,
                    optimizer=optim,
                    device=self.device,
                    criterion=self._loss_function,
                )

            # Meta-testing testing (using the test-test set).
            self.test_test_loader.sampler.set_active_tasks(new_tasks)
            results = evaluate_model(
                model=self.model,
                loader=self.test_test_loader,
                device=self.device,
                criterion=self._loss_function,
            )
            correct = results["total_correct"]

            acc = correct / len(self.test_test_loader.sampler.indices)
            meta_test_test_accuracies.append(acc)

            # Meta-testing testing (using the test-train set).
            self.test_train_loader.sampler.set_active_tasks(new_tasks)
            results = evaluate_model(
                model=self.model.module,
                loader=self.test_train_loader,
                device=self.device,
                criterion=self._loss_function,
            )
            correct = results["total_correct"]

            acc = correct / len(self.test_train_loader.sampler.indices)
            meta_test_train_accuracies.append(acc)

        return meta_test_train_accuracies, meta_test_test_accuracies

    def get_named_test_train_params(self):
        """Filter out the params from test_train_param_names."""
        model = self.get_model()
        named_test_train_params = {}
        for n, p in model.named_parameters():
            if n in self.test_train_param_names:
                named_test_train_params[n] = p

        return named_test_train_params

    def get_named_output_params(self):
        """Filter out the params from output_param_names."""
        model = self.get_model()
        named_output_params = {}
        for n, p in model.named_parameters():
            if n in self.output_param_names:
                named_output_params[n] = p

        return named_output_params

    def reset_params(self, params):
        """Helper function to reinitialize params."""
        for param in params:
            if len(param.shape) > 1:
                kaiming_normal_(param)
            else:
                zeros_(param)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("OML meta-testing setup")
        eo["post_epoch"].append("Run meta testing phase")
        eo["create_loaders"].append("Create loaders for the meta testing phase")

        return eo


class ResetOMLTaskParams(object):
    """
    Reset the task parameters, within the output layer weights,
    prior to training over those tasks. The mixin must be used with
    the `OMLNetwork`.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)
        model = self.get_model()
        assert isinstance(model, OMLNetwork)

    def pre_task(self, tasks):
        super().pre_task(tasks)
        model = self.get_model()
        for t in tasks:
            task_weights = model.adaptation[0].weight[t, :].unsqueeze(0)
            nn.init.kaiming_normal_(task_weights)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "ResetOMLTaskParams"
        eo["setup_experiment"].append(name + ": Ensure use of OMLNetwork.")
        eo["pre_task"].append(name + ": Reset the output params for upcoming tasks.")
        return eo
