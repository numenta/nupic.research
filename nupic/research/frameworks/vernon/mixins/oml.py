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
from scipy import stats
from torch.nn.init import kaiming_normal_, zeros_
from torch.optim import Adam

from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model


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
            - reset_fast_params: whether to reset (i.e. re-init) the fast
                                 params prior to meta-test training
            - lr_sweep_range: list of learning rates to attempt meta-test training.
                              The best one, according to the meta-test test set,
                              will be chosen and used for the meta-testing phase.

        """
        self.run_meta_test = config.get("run_meta_test", False)
        self.reset_fast_params = config.get("reset_fast_params", True)
        self.lr_sweep_range = config.get("lr_sweep_range", [1e-1, 1e-2, 1e-3, 1e-4])
        super().setup_experiment(config)

    def create_loaders(self, config):
        super().create_loaders(config)

        # Only initialize test loader if needed
        if not self.run_meta_test:
            return

        eval_set = self.load_dataset(config, train=False)

        self.test_train_loader = self.create_train_dataloader(config, eval_set)
        self.test_test_loader = self.create_validation_dataloader(config, eval_set)

        self.num_classes_eval = min(
            config.get("num_classes_eval", 50),
            self.test_train_loader.sampler.num_classes
        )

    def post_epoch(self):
        super().post_epoch()
        if self.should_stop() and self.run_meta_test:
            # meta-training phase complete, perform meta-testing phase
            for num_classes_learned in [10, 50, 100, 200, 600]:
                if num_classes_learned > self.num_classes_eval:
                    break

                # TODO log results in addition to simply priting them to stdout
                results = self.run_meta_testing_phase(num_classes_learned)
                test_train_accs, test_test_accs = results
                print("Accuracy for meta-testing phase over"
                      f" {num_classes_learned} num classes.")
                print("Meta-test training accuracies:", test_train_accs)
                print("Meta-test tests accuracies:", test_test_accs)

    def find_best_lr(self, num_classes_learned):
        """Adapted from original OML repo"""

        lr_all = []

        # Grid search over lr
        for _lr_search_runs in range(0, 5):

            # Choose num_classes_learned random classes from the non-background set
            # to train on.
            new_tasks = np.random.choice(
                self.num_classes_eval, num_classes_learned, replace=False
            )

            max_acc = -1000
            for lr in self.lr_sweep_range:

                # Reset fast weights.
                named_params = dict(self.get_named_fast_params())
                params = list(named_params.values())
                if self.reset_fast_params:
                    self.reset_params(params)

                optim = Adam(params, lr=lr)
                for task in new_tasks:
                    self.test_train_loader.sampler.set_active_tasks(task)
                    train_model(
                        model=self.model.module,
                        loader=self.test_train_loader,
                        optimizer=optim,
                        device=self.device,
                        criterion=self._loss_function,
                    )

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
        """Adapted from original OML repo"""

        lr = self.find_best_lr(num_classes_learned)
        print(f"Found best lr={lr} for num_classes_learned={num_classes_learned}")

        meta_test_test_accuracies = []
        meta_test_train_accuracies = []
        for _current_run in range(0, 15):

            # Choose num_classes_learned random classes from the non-background set to
            # test on
            new_tasks = np.random.choice(
                self.num_classes_eval, num_classes_learned, replace=False
            )

            # Reset fast weights.
            named_params = dict(self.get_named_fast_params())
            params = list(named_params.values())
            if self.reset_fast_params:
                self.reset_params(params)

            # meta-testing training
            optim = Adam(params, lr=lr)
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

    def reset_params(self, params):
        for param in params:
            if len(param.shape) > 1:
                kaiming_normal_(param)
            else:
                zeros_(param)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].insert(0, "OML additional attributes")
        eo["post_epoch"].append("Run meta testing phase")
        eo["create_loaders"].append("Create loaders for the meta testing phase")

        return eo
