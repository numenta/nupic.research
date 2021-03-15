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
from pandas import DataFrame


class BaseLogger:
    def __init__(self, model, config=None):
        defaults = dict(debug_weights=False, verbose=0)
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.model = model
        self.log = {}

    def log_pre_epoch(self):
        # reset log
        self.log = {}

    def log_post_epoch(self):
        if self.verbose > 0:
            print(self.log)
        if self.debug_weights:
            self.log_weights()

    def log_pre_batch(self):
        pass

    def log_post_batch(self):
        pass

    def log_metrics(self, loss, acc, train, noise):

        if train:
            self.log["train_loss"] = loss
            self.log["train_acc"] = acc
            if self.model.lr_scheduler:
                self.log["learning_rate"] = self.model.lr_scheduler.get_lr()[0]
            else:
                self.log["learning_rate"] = self.model.learning_rate
        else:
            if noise:
                self.log["noise_loss"] = loss
                self.log["noise_acc"] = acc
            else:
                self.log["val_loss"] = loss
                self.log["val_acc"] = acc

        if train and self.debug_weights:
            self.log_weights()

    def log_weights(self):
        """Log weights for all layers which have params."""
        if "param_layers" not in self.model.__dict__:
            self.model.param_layers = defaultdict(list)
            for m, ltype in [
                (m, self.model.has_params(m)) for m in self.model.network.modules()
            ]:
                if ltype:
                    self.model.param_layers[ltype].append(m)

        # log stats (mean and weight instead of standard distribution)
        for ltype, layers in self.model.param_layers.items():
            for idx, m in enumerate(layers):
                # keep track of mean and std of weights
                self.log[ltype + "_" + str(idx) + "_mean"] = torch.mean(m.weight).item()
                self.log[ltype + "_" + str(idx) + "_std"] = torch.std(m.weight).item()


class SparseLogger(BaseLogger):
    def __init__(self, model, config=None):
        super().__init__(model, config)
        defaults = dict(
            log_magnitude_vs_coactivations=False,  # scatter plot of magn. vs coacts.
            debug_sparse=False,
            log_sparse_layers_grid=False,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.model = model

    def log_metrics(self, loss, acc, train, noise):
        super().log_metrics(loss, acc, train, noise)
        if train and self.debug_sparse:
            self._log_sparse_levels()
        if train and self.log_magnitude_vs_coactivations:
            self._log_magnitude_and_coactivations(train)

    def _log_magnitude_and_coactivations(self, train):

        for i, module in enumerate(self.model.sparse_modules):

            m = module.m
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
            seaborn_config = dict(rc={"figure.figsize": (11.7, 8.27)}, style="white")

            self.log["scatter_mag_vs_coacts_layer-{}".format(str(i))] = dict(
                data=dataframe, x=x, y=y, hue=hue, seaborn_config=seaborn_config
            )

    def _log_sparse_levels(self):
        with torch.no_grad():
            for idx, module in enumerate(self.model.sparse_modules):
                zero_mask = module.m.weight == 0
                zero_count = torch.sum(zero_mask.int()).item()
                size = np.prod(module.shape)
                log_name = "sparse_level_l" + str(idx)
                self.log[log_name] = 1 - zero_count / size

                # log image as well
                if self.log_sparse_layers_grid:
                    if self.model.has_params(module.m) == "conv":
                        ratio = 255 / np.prod(module.shape[2:])
                        heatmap = (
                            torch.sum(module.m.weight, dim=[2, 3]).float() * ratio
                        ).int()
                        self.log["img_" + log_name] = heatmap.tolist()


class DSNNLogger(SparseLogger):
    def __init__(self, model, config=None):
        super().__init__(model, config)
        defaults = dict(log_surviving_synapses=False, log_masks=False)
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.model = model

    def save_masks(
        self,
        idx,
        new_mask,
        keep_mask,
        add_mask,
        num_add,
        hebbian_mask=None,
        magnitude_mask=None,
    ):
        """Log different masks in DSNN"""

        if self.log_masks:
            num_synapses = np.prod(new_mask.shape)
            self.log["keep_mask_l" + str(idx)] = (
                torch.sum(keep_mask).item() / num_synapses
            )
            self.log["add_mask_l" + str(idx)] = (
                torch.sum(add_mask).item() / num_synapses
            )
            self.log["new_mask_l" + str(idx)] = (
                torch.sum(new_mask).item() / num_synapses
            )
            self.log["missing_weights_l" + str(idx)] = num_add / num_synapses

            # conditional logs
            if hebbian_mask is not None:
                self.log["hebbian_mask_l" + str(idx)] = (
                    torch.sum(hebbian_mask).item() / num_synapses
                )
            if magnitude_mask is not None:
                self.log["magnitude_mask_l" + str(idx)] = (
                    torch.sum(magnitude_mask).item() / num_synapses
                )

    def save_surviving_synapses(self, module, keep_mask, add_mask):
        """Tracks added and surviving synapses"""

        self.survival_ratios = []
        if self.log_surviving_synapses and self.model.pruning_active:
            self.survival_ratios = []
            # count how many synapses from last round have survived
            if module.added_synapses is not None:
                total_added = torch.sum(module.added_synapses).item()
                surviving = torch.sum(module.added_synapses & keep_mask).item()
                if total_added:
                    survival_ratio = surviving / total_added
                    self.survival_ratios.append(survival_ratio)

                # keep track of new synapses to count surviving on next round
                module.added_synapses = add_mask
                self.log["mask_sizes_l" + str(module.pos)] = module.nonzero_params()
                self.log["surviving_synapses_l" + str(module.pos)] = survival_ratio

    def log_post_epoch(self):
        super().log_post_epoch()
        # adds tracking of average surviving synapses
        if self.log_surviving_synapses and self.model.pruning_active:
            self.log["surviving_synapses_avg"] = np.mean(self.survival_ratios)
