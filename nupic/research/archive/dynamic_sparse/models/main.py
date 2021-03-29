# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import json
import os
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers

from nupic.research.frameworks.dynamic_sparse.networks import NumScheduler
from nupic.torch.modules import update_boost_strength

from .loggers import BaseLogger, SparseLogger
from .modules import SparseModule

__all__ = ["BaseModel", "SparseModel"]


class BaseModel:
    """Base model, with training loops and logging functions."""

    def __init__(self, network=None, config=None):

        defaults = dict(
            optim_alg="SGD",
            learning_rate=0.1,
            momentum=0.9,
            nesterov_momentum=False,
            device="cpu",
            lr_scheduler=False,
            lr_step_size=1,
            pruning_interval=1,
            weight_prune_perc=0,
            grad_prune_perc=0,
            test_noise=False,
            weight_decay=1e-4,
            use_multiple_gpus=False,
            train_batches_per_epoch=np.inf,  # default - don't limit the batches
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)

        # save config to restore the model later
        self.config = config
        self.device = torch.device(self.device)
        if self.use_multiple_gpus:
            network = nn.DataParallel(network)
        self.network = network.to(self.device)
        self.config = deepcopy(config)

    def setup(self, config=None):

        # allow setup to receive a new config
        if config is not None:
            self.__dict__.update(config)

        # init optimizer
        if self.optim_alg == "Adam":
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim_alg == "SGD":
            # added weight decay
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov_momentum,
            )

        # add a learning rate scheduler
        if self.lr_scheduler == "MultiStepLR":
            self.lr_scheduler = schedulers.MultiStepLR(
                self.optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma
            )
        elif self.lr_scheduler == "StepLR":
            self.lr_scheduler = schedulers.StepLR(
                self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
            )

        # init loss function
        self.loss_func = nn.CrossEntropyLoss()

        # init batch info per epic.
        self._make_attr_schedulable("train_batches_per_epoch")

        self.logger = BaseLogger(self, config=self.config)

    def run_epoch(self, dataset, epoch, test_noise_local=False):
        self.current_epoch = epoch + 1
        self.logger.log_pre_epoch()
        self._pre_epoch_setup()
        self.network.train()
        self._run_one_pass(dataset.train_loader, train=True)
        self.network.eval()
        self._run_one_pass(dataset.test_loader, train=False)
        # run one additional testing epoch for noise
        if self.test_noise or test_noise_local:
            self._run_one_pass(dataset.noise_loader, train=False, noise=True)
        self._post_epoch_updates(dataset)
        self.logger.log_post_epoch()

        return self.logger.log

    def _make_attr_schedulable(self, attr):

        value = getattr(self, attr)
        if isinstance(value, NumScheduler):
            return

        if not isinstance(value, Iterable):
            value = [value]
        setattr(self, attr, NumScheduler(value))

    def _pre_epoch_setup(self):
        pass

    def _post_epoch_updates(self, dataset=None):
        # update learning rate
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.network.apply(update_boost_strength)

        # iterate num_schedulers
        for val in self.__dict__.values():
            if isinstance(val, NumScheduler):
                val.step()

        self.logger.log_post_epoch()

    def _post_optimize_updates(self):
        pass

    def _run_one_pass(self, loader, train=True, noise=False):
        epoch_loss = 0
        correct = 0
        for idx, (inputs, targets) in enumerate(loader):
            self.logger.log_pre_batch()
            # Limit number of batches per epoch if desired.
            if train:
                if idx >= self.train_batches_per_epoch.get_value():
                    break
            # setup for training
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            # training loop
            with torch.set_grad_enabled(train):
                # forward + backward + optimize
                outputs = self.network(inputs)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(targets == preds).item()
                loss = self.loss_func(outputs, targets)
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self._post_optimize_updates()

            # keep track of loss
            epoch_loss += loss.item() * inputs.size(0)
            self.logger.log_post_batch()

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = correct / len(loader.dataset)
        self.logger.log_metrics(loss, acc, train, noise)

    def has_params(self, module):
        if isinstance(module, nn.Linear):
            return "linear"
        elif isinstance(module, nn.Conv2d) and not self.sparse_linear_only:
            return "conv"

    def save(self, checkpoint_dir, experiment_name):
        """
        Save the model in this directory.
        :param checkpoint_dir:
        """
        # experiment_root = os.path.join(checkpoint_dir, experiment_name)
        # if not os.path.exists(experiment_root):
        #     os.mkdir(experiment_root)

        checkpoint_path = os.path.join(checkpoint_dir, experiment_name + ".pth")
        torch.save(self.network.state_dict(), checkpoint_path)

        config_file = os.path.join(checkpoint_dir, experiment_name + ".json")
        with open(config_file, "w") as config_handler:
            json.dump(self.config, config_handler)

        return

    def restore(self, checkpoint_dir, experiment_name):
        """
        :param checkpoint_path: Loads model from this checkpoint path.
        If path is a directory, will append the parameter model_filename
        """
        # print("loading from", checkpoint_path)
        # experiment_root = os.path.join(checkpoint_dir, experiment_name)
        checkpoint_path = os.path.join(checkpoint_dir, experiment_name + ".pth")
        self.network.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    def evaluate_noise(self, dataset):
        """External function used to evaluate noise on pre-trained models"""
        self.network.eval()
        self._run_one_pass(dataset.noise_loader, train=False, noise=True)
        loss, acc = self.logger.log["noise_loss"], self.logger.log["noise_acc"]
        return loss, acc

    def calculate_num_params(self):
        total_params = 0
        zero_params = 0
        for m in self.network.modules():
            if self.has_params(m):
                total_params += np.prod(m.weight.data.shape)
                zero_params += torch.sum((m.weight.data == 0).float()).item()
                if m.bias is not None:
                    total_params += np.prod(m.bias.data.shape)
                    zero_params += torch.sum((m.bias.data == 0).float()).item()

        return total_params, zero_params

    def train(self, dataset, num_epochs, test_noise=False):
        """
        Added method to allow running the class outside Ray
        Print accuracy to screen
        Return log file over epochs for analysis, with all elements
        """
        results = defaultdict(list)
        for epoch in range(num_epochs):
            # regular call function, as in Ray. Added option to evaluate test noise
            log = self.run_epoch(dataset, epoch, test_noise)
            # print intermediate results
            if test_noise:
                print(
                    "Train acc: {:.4f}, Val acc: {:.4f}, Noise acc: {:.4f}".format(
                        log["train_acc"], log["val_acc"], log["noise_acc"]
                    )
                )
            else:
                print(
                    "Train acc: {:.4f}, Val acc: {:.4f}".format(
                        log["train_acc"], log["val_acc"]
                    )
                )
            # add log to results
            for var in log:
                results[var].append(log[var])

        return results


class SparseModel(BaseModel):
    """Sparsity implemented by:
    - Masking on the weights
    - Zeroing out gradients in backprop before optimizer steps
    """

    def setup(self, config=None):
        super(SparseModel, self).setup(config)

        # add specific defaults
        new_defaults = dict(
            start_sparse=None,
            end_sparse=None,
            sparse_linear_only=False,
            on_perc=0.1,
            epsilon=None,
            sparse_type="precise",  # precise, precise_per_output, approximate
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # define all modules (those that are sparsifiable)
        self.sparse_modules = self._get_sparse_modules()

        # option to define sparsity by on_perc instead of epsilon
        if "on_perc" in self.__dict__:
            self._make_attr_iterable("on_perc", self.sparse_modules)
            self.epsilon = None

        # define sparse modules
        for idx, module in enumerate(self.sparse_modules):
            # define on_perc
            if self.epsilon:
                shape = module.shape
                module.on_perc = self.epsilon * np.sum(shape) / np.prod(shape)
            else:
                module.on_perc = self.on_perc[idx]
            module.create_mask(self.sparse_type)
            with torch.no_grad():
                module.apply_mask()
            module.save_num_params()

        self.logger = SparseLogger(self, config=self.config)

    def _sparse_module_type(self):
        return SparseModule

    def _post_optimize_updates(self):
        # zero out the weights after the step - avoid propagating bias
        with torch.no_grad():
            for module in self.sparse_modules:
                module.apply_mask()

    def _is_sparsifiable(self, module):
        return isinstance(module, nn.Linear) or (
            isinstance(module, nn.Conv2d) and not self.sparse_linear_only
        )

    def _get_sparse_modules(self):
        sparse_modules = []
        module_type = self._sparse_module_type()
        for idx, m in enumerate(
            list(self.network.modules())[self.start_sparse : self.end_sparse]
        ):
            if self._is_sparsifiable(m):
                sparse_modules.append(module_type(m=m, device=self.device, pos=idx))

        return sparse_modules

    def _make_attr_iterable(self, attr, counterpart):
        """
        This function (called in setup), ensures that a pre-existing attr
        in an iterable (list) of length equal to self.sparse_modules.

        :param attr: str - name of attribute to make into iterable
        :param counterpart: Iterable with defined length - determines how many times
                            to repeat the value of 'attr'. Defaults to
                            self.sparse_modules.
        """
        value = getattr(self, attr)
        if isinstance(value, Iterable):
            assert len(value) == len(
                counterpart
            ), """
                Expected "{}" to be of same length as counterpart ({}).
                Got {} of type {}.
                """.format(
                attr, len(counterpart), value, type(value)
            )
        else:
            value = [value] * len(counterpart)
            setattr(self, attr, value)
