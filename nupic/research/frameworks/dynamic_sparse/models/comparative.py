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

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers


class BaseModel:
    """Base model, with training loops and logging functions."""

    def __init__(self, network, config=None):
        defaults = dict(
            optim_alg="SGD",
            learning_rate=0.1,
            momentum=0.9,
            device="cpu",
            lr_scheduler=False,
            debug_sparse=False,
            debug_weights=False,
            start_sparse=None,
            end_sparse=None,
            pruning_interval=1,
            log_images=False,
            flip=False,
            weight_prune_perc=0,
            grad_prune_perc=0,
            test_noise=False,
            percent_on=0.3,
            boost_strength=1.4,
            boost_strength_factor=0.7,
            weight_decay=1e-4,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)

        # init remaining
        self.device = torch.device(self.device)
        self.network = network.to(self.device)

    def setup(self):

        # init optimizer
        if self.optim_alg == "Adam":
            self.optimizer = optim.Adam(
                self.network.parameters(), lr=self.learning_rate
            )
        elif self.optim_alg == "SGD":
            # added weight decay
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )

        # add a learning rate scheduler
        if self.lr_scheduler:
            self.lr_scheduler = schedulers.MultiStepLR(
                self.optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma
            )

        # init loss function
        self.loss_func = nn.CrossEntropyLoss()

    def run_epoch(self, dataset, epoch):
        self.current_epoch = epoch + 1
        self.log = {}
        self.network.train()
        self._run_one_pass(dataset.train_loader, train=True)
        self.network.eval()
        self._run_one_pass(dataset.test_loader, train=False)
        self._post_epoch_updates(dataset)

        return self.log

    def _post_epoch_updates(self, dataset=None):

        # run one additional testing epoch for noise
        if self.test_noise:
            self._run_one_pass(dataset.noise_loader, train=False, noise=True)

        # update learning rate
        if self.lr_scheduler:
            self.lr_scheduler.step()

    def _run_one_pass(self, loader, train=True, noise=False):
        epoch_loss = 0
        correct = 0
        for inputs, targets in loader:
            # setup for training
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
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

            # keep track of loss
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = correct / len(loader.dataset)
        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        if train and self.debug_weights:
            self._log_weights()

    @staticmethod
    def has_params(module):
        if isinstance(module, nn.Linear):
            return "linear"
        elif isinstance(module, nn.Conv2d):
            return "conv"

    def _log_weights(self):
        """Log weights for all layers which have params."""
        if "param_layers" not in self.__dict__:
            self.param_layers = defaultdict(list)
            for m, ltype in [(m, self.has_params(m)) for m in self.network.modules()]:
                if ltype:
                    self.param_layers[ltype].append(m)

        # log stats (mean and weight instead of standard distribution)
        for ltype, layers in self.param_layers.items():
            for idx, m in enumerate(layers):
                # keep track of mean and std of weights
                self.log[ltype + "_" + str(idx) + "_mean"] = torch.mean(m.weight).item()
                self.log[ltype + "_" + str(idx) + "_std"] = torch.std(m.weight).item()

    def save(self):
        pass

    def restore(self):
        pass


class SparseModel(BaseModel):
    """Sparsity implemented by:
    - Masking on the weights. UPDATE: Not based on epsilon, but fixed on_perc
    - Zeroing out gradients in backprop before optimizer steps
    """

    def setup(self):
        super().setup()

        # add specific defaults
        if "on_perc" not in self.__dict__:
            self.on_perc = 0.1

        with torch.no_grad():
            # calculate sparsity masks
            self.masks = []
            self.num_params = []  # added for paper implementation

            # define sparse modules
            self.sparse_modules = []
            for m in list(self.network.modules())[self.start_sparse : self.end_sparse]:
                if self.has_params(m):
                    self.sparse_modules.append(m)

            # initialize masks
            for m in self.sparse_modules:
                shape = m.weight.shape
                mask = (torch.rand(shape) < self.on_perc).float().to(self.device)
                m.weight.data *= mask
                self.masks.append(mask)
                self.num_params.append(torch.sum(mask).item())

    def _run_one_pass(self, loader, train, noise=False):
        """TODO: reimplement by calling super and passing a hook"""
        epoch_loss = 0
        epoch_correct = 0
        for inputs, targets in loader:
            # setup for training
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()

            # forward + backward + optimize
            with torch.set_grad_enabled(train):
                outputs = self.network(inputs)
                _, preds = torch.max(outputs, 1)
                epoch_correct += torch.sum(targets == preds).item()
                loss = self.loss_func(outputs, targets)
                if train:
                    loss.backward()
                    # zero the gradients for dead connections
                    for mask, m in zip(self.masks, self.sparse_modules):
                        m.weight.grad *= mask
                    self.optimizer.step()

            # keep track of loss and accuracy
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = epoch_correct / len(loader.dataset)
        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        if train and self.debug_weights:
            self._log_weights()

        # add monitoring of sparse levels
        if train and self.debug_sparse:
            self._log_sparse_levels()

    def _log_sparse_levels(self):
        with torch.no_grad():
            for idx, m in enumerate(self.sparse_modules):
                zero_mask = m.weight == 0
                zero_count = torch.sum(zero_mask.int()).item()
                size = np.prod(m.weight.shape)
                log_name = "sparse_level_l" + str(idx)
                self.log[log_name] = 1 - zero_count / size

                # log image as well
                if self.log_images:
                    if self.has_params(m) == "conv":
                        ratio = 255 / np.prod(m.weight.shape[2:])
                        heatmap = (
                            torch.sum(m.weight, dim=[2, 3]).float() * ratio
                        ).int()
                        self.log["img_" + log_name] = heatmap.tolist()


class DynamicRep(SparseModel):
    """REMINDER: need to remove downsampling layers - figure out how to do"""

    def setup(self):
        super().setup()
        # define sparsity - more intuitive than on_perc
        self.h_tolerance = 0.05
        self.zeta = 0.2
        # count params
        self._initialize_prune_threshold()
        self._count_params()

        # debugging start
        toprune_params = int(self.available_params * (self.zeta))
        toprune_baseline = (
            int(toprune_params * (1 - self.h_tolerance)),
            int(toprune_params * (1 + self.h_tolerance)),
        )
        print(toprune_params)
        print(toprune_baseline)

        # initialize data structure keep track of added synapses
        self.added_synapses = [None for m in self.masks]

    def _count_params(self):
        """
        Count parameters of the network (sparse modules alone)
        No need to keep track of full parameters, just the available ones
        """
        self.available_params = 0
        for m in list(self.network.modules())[self.start_sparse : self.end_sparse]:
            if self.has_params(m):
                self.available_params += torch.sum(m.weight != 0).item()

    def _initialize_prune_threshold(self):
        """Initialize prune threshold h"""
        weighted_mean = 0
        total_params_count = 0
        # initialize h and total_params
        with torch.no_grad():
            for m in list(self.network.modules())[self.start_sparse : self.end_sparse]:
                if self.has_params(m):
                    # count how many weights are not equal to 0
                    count_p = torch.sum(m.weight != 0).item()
                    # get topk for that level, and weight by num of values
                    non_zero = torch.abs(m.weight[m.weight != 0]).view(-1)
                    val, _ = torch.kthvalue(non_zero, int(len(non_zero) * self.on_perc))
                    weighted_mean += count_p * val.item()
                    total_params_count += count_p

        # get initial value for h based on enforced sparsity
        self.h = weighted_mean / total_params_count
        print(self.h)

    def _run_one_pass(self, loader, train, noise=False):
        super()._run_one_pass(loader, train, noise)
        if train:
            self.reinitialize_weights()

    def reinitialize_weights(self):
        """Steps
        1- calculate how many weights should go to the layer
        2- call prune on each layer
        3- update H
        """
        with torch.no_grad():
            total_available = self.available_params
            # count to prune based on a fixed percentage of the surviving weights
            toprune_count = int(self.zeta * total_available)
            self.pruned_count = 0
            self.grown_count = 0
            for idx, m in enumerate(self.sparse_modules):
                # calculate number of weights to add
                available = torch.sum(m.weight != 0).item()
                num_add = int(available / total_available * toprune_count)
                # prune weights
                new_mask, keep_mask, grow_mask = self.prune_and_grow(
                    m.weight.clone().detach(), num_add
                )
                self.masks[idx] = new_mask.float()
                m.weight.data *= self.masks[idx].float()

                # DEBUGGING STUFF. TODO: move code to some other place

                # count how many synapses from last round have survived
                if self.added_synapses[idx] is not None:
                    total_added = torch.sum(self.added_synapses[idx]).item()
                    surviving = torch.sum(self.added_synapses[idx] & keep_mask).item()
                    if total_added:
                        survival_ratio = surviving / total_added
                        # log if in debug sparse mode
                        if self.debug_sparse:
                            self.log["surviving_synapses_l" + str(idx)] = survival_ratio

                # keep track of new synapses to count surviving on next round
                self.added_synapses[idx] = grow_mask

        # track main parameters
        self.log["dyre_total_available"] = total_available
        self.log["dyre_delta_available"] = self.available_params - total_available
        self.log["dyre_to_prune"] = toprune_count
        self.log["dyre_pruned_count"] = self.pruned_count
        self.log["dyre_grown_count"] = self.grown_count
        self.log["dyre_h_threshold"] = self.h
        # update H according to the simple rule
        # if actual pruned less than threshold, reduce tolerance
        if self.pruned_count < int(toprune_count * (1 - self.h_tolerance)):
            self.h *= 2
            self.log["dyre_h_delta"] = 1
        # if greater, increase tolerance
        elif self.pruned_count > int(toprune_count * (1 + self.h_tolerance)):
            self.h /= 2
            self.log["dyre_h_delta"] = -1

        # keep track of mask sizes when debugging
        if self.debug_sparse:
            for idx, m in enumerate(self.masks):
                self.log["mask_sizes_l" + str(idx)] = torch.sum(m).item()

    def prune_and_grow(self, weight, num_add):
        """Steps
        1- Sample positions to grow new weights (sample indexes)
        2- Update parameter count
        3- Prune parameters based on H - update prune mask
        4- Grow randomly sampled parameters - update add mask
        5 - return both masks
        """
        with torch.no_grad():

            # GROW
            # identify non-zero entries
            nonzero_idxs = torch.nonzero(weight)
            num_available_params = len(nonzero_idxs)
            # randomly sample
            random_sample = torch.randperm(len(nonzero_idxs))[:num_add]
            togrow_idxs = nonzero_idxs[random_sample]
            # update the mask with added weights
            grow_mask = torch.zeros(weight.shape, dtype=torch.bool).to(self.device)
            grow_mask[tuple(togrow_idxs.T)] = 1
            num_grow = len(togrow_idxs)

            # PRUNE

            keep_mask = (torch.abs(weight) > self.h).to(self.device)
            num_prune = num_available_params - torch.sum(keep_mask).item()

            # combine both
            new_mask = keep_mask | grow_mask

            # update parameter count
            self.pruned_count += num_prune
            self.grown_count += num_grow
            self.available_params += num_grow
            self.available_params -= num_prune

        # track added connections
        return new_mask, keep_mask, grow_mask
