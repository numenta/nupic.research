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
from collections.abc import Iterable
from collections import namedtuple
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as schedulers
from pandas import DataFrame

from nupic.research.frameworks.dynamic_sparse.networks import (
    DynamicSparseBase,
    NumScheduler,
    init_coactivation_tracking,
)
from nupic.research.frameworks.dynamic_sparse.networks.layers import (
    init_coactivation_tracking,
)
from nupic.torch.modules import update_boost_strength

class BaseModel:
    """Base model, with training loops and logging functions."""

    def __init__(self, network, config=None):
        defaults = dict(
            optim_alg="SGD",
            learning_rate=0.1,
            momentum=0.9,
            device="cpu",
            lr_scheduler=False,
            lr_step_size=1,
            debug_sparse=False,
            debug_weights=False,
            pruning_interval=1,
            log_images=False,
            weight_prune_perc=0,
            grad_prune_perc=0,
            test_noise=False,
            weight_decay=1e-4,
            sparse_linear_only=False,
            epsilon=None,
            sparse_type='precise', # precise, precise_per_output, approximate
            verbose=0,
            train_batches_per_epoch=np.inf,  # default - don't limit the batches
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

    def run_epoch(self, dataset, epoch, test_noise_local=False):
        self.current_epoch = epoch + 1
        self.log = {}
        self._pre_epoch_setup()
        self.network.train()
        self._run_one_pass(dataset.train_loader, train=True)
        self.network.eval()
        self._run_one_pass(dataset.test_loader, train=False)
        # run one additional testing epoch for noise
        if self.test_noise or test_noise_local:
            self._run_one_pass(dataset.noise_loader, train=False, noise=True)
        self._post_epoch_updates(dataset)
        if self.verbose > 0:
            print(self.log)

        return self.log

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

    def _post_optimize_updates(self):
        pass

    def _run_one_pass(self, loader, train=True, noise=False):
        epoch_loss = 0
        correct = 0
        for idx, (inputs, targets) in enumerate(loader):

            # Limit number of batches per epoch if desired.
            if train:
                if idx >= self.train_batches_per_epoch.get_value():
                    break

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
                    self._post_optimize_updates()

            # keep track of loss
            epoch_loss += loss.item() * inputs.size(0)

        # store loss and acc at each pass
        loss = epoch_loss / len(loader.dataset)
        acc = correct / len(loader.dataset)
        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
            if self.lr_scheduler:
                self.log["learning_rate"] = self.lr_scheduler.get_lr()[0]
            else:
                self.log["learning_rate"] = self.learning_rate
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        if train and self.debug_weights:
            self._log_weights()

    def has_params(self, module):
        if isinstance(module, nn.Linear):
            return "linear"
        elif isinstance(module, nn.Conv2d) and not self.sparse_linear_only:
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

    def save(self, checkpoint_dir):
        # TODO: Implement
        pass

    def restore(self, checkpoint_dir):
        # TODO: Implement
        pass

    # def _save(self, checkpoint_dir):
    #     return self.model_save(checkpoint_dir)

    # def _restore(self, checkpoint):
    #     """Subclasses should override this to implement restore().

    #     Args:
    #         checkpoint (str | dict): Value as returned by `_save`.
    #             If a string, then it is the checkpoint path.
    #     """
    #     self.model_restore(checkpoint)

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

    def _post_optimize_updates(self):
        # zero out the weights after the step - avoid propagating bias
        with torch.no_grad():
            for module in self.sparse_modules:
                module.apply_mask()

    def _is_sparsifiable(self, module):
        return (isinstance(module, nn.Linear) or 
            (isinstance(module, nn.Conv2d) and not self.sparse_linear_only))

    def _get_sparse_modules(self):
        sparse_modules = []
        for idx, m in enumerate(list(self.network.modules())[self.start_sparse : self.end_sparse]):
            if self._is_sparsifiable(m):
                sparse_modules.append(SparseModule(m=m, device=self.device, pos=idx))

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
                attr, counterpart, value, type(value)
            )
        else:
            value = [value] * len(counterpart)
            setattr(self, attr, value)

    def setup(self):
        super(SparseModel, self).setup()

        # add specific defaults
        new_defaults = dict(
            start_sparse=None,
            end_sparse=None,
            sparse_linear_only=False,
            init_sparse_fixed=True,
            init_sparse_fixed_per_output=False,
            log_magnitude_vs_coactivations=False,  # scatter plot of magn. vs coacts.
            reset_coactivations=True,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # calculate sparsity masks
        self.masks = []
        self.num_params = []  # added for paper implementation

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
                module.on_perc = self.epsilon * np.sum(shape) / np.prod(shape)
            else:
                module.on_perc = self.on_perc[idx]
            if module.on_perc < 1:
                module.create_mask(self.sparse_type)
                with torch.no_grad():        
                    module.apply_mask()
                    module.save_num_params()

        # TODO: remove logging to separate class
        # if self.log_magnitude_vs_coactivations:
        #     self.network.apply(init_coactivation_tracking)

    def _run_one_pass(self, loader, train, noise=False):
        """TODO: remove logging to separate class"""
        super()._run_one_pass(loader, train, noise)

        # save scatter plot of magn vs coacts.
        if train and self.log_magnitude_vs_coactivations:
            for i, m in enumerate(self.sparse_modules):

                coacts = m.coactivations.clone().detach().to("cpu").numpy()
                weight = m.weight.clone().detach().to("cpu").numpy()
                grads = m.weight.grad.clone().detach().to("cpu").numpy()

                mask = ((m.weight.grad != 0)).to("cpu").numpy()

                coacts = coacts[mask]
                weight = weight[mask]
                grads = grads[mask]
                grads = np.log(np.abs(grads))

                x, y, hue = "coactivations", "weight", "log_abs_grads"

                dataframe = DataFrame(
                    {x: coacts.flatten(), y: weight.flatten(), hue: grads.flatten()}
                )
                seaborn_config = dict(
                    rc={"figure.figsize": (11.7, 8.27)}, style="white"
                )

                self.log["scatter_mag_vs_coacts_layer-{}".format(str(i))] = dict(
                    data=dataframe, x=x, y=y, hue=hue, seaborn_config=seaborn_config
                )

        if train and self.debug_weights:
            self._log_weights()

        # add monitoring of sparse levels
        if train and self.debug_sparse:
            self._log_sparse_levels()

    def _log_sparse_levels(self):
        with torch.no_grad():
            for idx, module in enumerate(self.sparse_modules):
                zero_mask = module.m.weight == 0
                zero_count = torch.sum(zero_mask.int()).item()
                size = np.prod(module.shape)
                log_name = "sparse_level_l" + str(idx)
                self.log[log_name] = 1 - zero_count / size

                # log image as well
                if self.log_images:
                    if self.has_params(m) == "conv":
                        ratio = 255 / np.prod(module.shape[2:])
                        heatmap = (
                            torch.sum(module.m.weight, dim=[2, 3]).float() * ratio
                        ).int()
                        self.log["img_" + log_name] = heatmap.tolist()

class SparseModule:
    """Module wrapper for sparse layers"""

    def __init__(self, m=None, pos=None, on_perc=None, num_params=None, mask=None, 
        weight_prune=None, hebbian_prune=None, device=None):
        """ document attributes """
        self.m = m
        self.pos = pos
        self.shape = m.weight.shape
        self.on_perc = on_perc
        self.num_params = num_params
        self.mask = mask
        self.hebbian_prune = hebbian_prune
        self.weight_prune = weight_prune
        self.device = device
        # logging
        self.added_synapses = None
        self.last_gradients = None

    def nonzero_params(self):
        return torch.sum(self.mask).item()

    def save_num_params(self):
        self.num_params = self.nonzero_params()

    def init_coactivation_tracking(self):
        if isinstance(self.m, DynamicSparseBase):
            self.m.apply(init_coactivation_tracking)
        else:
            self.m.coactivations = torch.zeros(self.m.weight.shape)

    def reset_coactivations(self, reset=True):
        if isinstance(self.m, DynamicSparseBase):
            self.m.reset_coactivations()
        else:
            self.m.coactivations[:] = 0

    def apply_mask(self):
        self.m.weight.data *= self.mask

    def create_mask(self, sparse_type):
        if sparse_type == 'precise':
            self._mask_fixed()
        elif sparse_type == 'precise_per_output':
            self._mask_fixed_per_output()
        elif sparse_type == 'approximate':
            self._mask_stochastic()
        else:
            raise ValueError("{} is an invalid option for sparse type. ".format(sparse_type))

    def _mask_fixed_per_output(self):
        """
        Similar implementation to how so dense
        Works in any number of dimension, considering the 1st one is the output
        """
        output_size = self.shape[0]
        input_size = np.prod(self.shape[1:])
        num_add = int(self.on_perc * input_size)
        mask = torch.zeros(self.shape, dtype=torch.bool)
        # loop over outputs
        for dim in range(output_size):
            # select a subsample of the indexes in the output unit
            all_idxs = np.array(list(product(*[range(s) for s in self.shape[1:]])))
            sampled_idxs = np.random.choice(
                range(len(all_idxs)), num_add, replace=False
            )
            selected = all_idxs[sampled_idxs]
            if len(selected) > 0:
                # change from long to wide format
                selected_wide = list(zip(*selected))
                # append the output dimension to wide format
                selected_wide.insert(0, tuple([dim] * len(selected_wide[0])))
                # apply the mask
                mask[selected_wide] = True

        self.mask = mask.float().to(self.device)

    def _mask_stochastic(self):
        """Sthocastic in num of params approach of sparsifying a tensor"""
        self.mask = (torch.rand(self.shape) < self.on_perc).float().to(self.device)

    def _mask_fixed(self):
        """
        Deterministic in number of params approach of sparsifying a tensor
        Sample N from all possible indices
        """
        all_idxs = np.array(list(product(*[range(s) for s in self.shape])))
        num_add = int(self.on_perc * np.prod(self.shape))
        sampled_idxs = np.random.choice(range(len(all_idxs)), num_add, replace=False)
        selected = all_idxs[sampled_idxs]

        mask = torch.zeros(self.shape, dtype=torch.bool)
        if len(selected.shape) > 1:
            mask[list(zip(*selected))] = True
        else:
            mask[selected] = True

        self.mask = mask.float().to(self.device)
