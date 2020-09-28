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

import torch

from nupic.torch.modules.prunable_sparse_weights import PrunableSparseWeightBase


class PruneLowMagnitude:
    """
    Prunes weights by magnitude at the beginning of select epochs.

    Each module of type PrunableSparseWeightBase will be pruned. Each of these
    modules must have attribute "_target_density".
    """
    def setup_experiment(self, config):
        """
        :param config:
            - prune_schedule: A list of (epoch, progress) pairs, where progress
                              denotes how far toward the target sparsity.
                              The progress is typically 1.0 at the end.
            - prune_curve_shape: Describes how to interpret the "progess" in
                                 prune_schedule. With "exponential", increasing
                                 progress at a fixed rate causes a fixed
                                 percentage if remaining weights to be pruned at
                                 each step. With "linear", a fixed percentage of
                                 the total weights will be pruned at each step.
        """
        super().setup_experiment(config)
        self.prune_schedule = dict(config["prune_schedule"])
        self.prune_curve_shape = config.get("prune_curve_shape", "exponential")

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch in self.prune_schedule:
            prune_progress = self.prune_schedule[self.current_epoch]

            for module in self.model.modules():
                if isinstance(module, PrunableSparseWeightBase):
                    if self.prune_curve_shape == "exponential":
                        density = module._target_density ** prune_progress
                    elif self.prune_curve_shape == "linear":
                        density = 1 - (
                            (1 - module._target_density) * prune_progress
                        )

                    mag = module.module.weight.detach().abs().view(-1)
                    on_indices = mag.topk(math.floor(mag.numel() * density))[1]
                    off_mask = torch.ones(module.zero_mask.shape,
                                          device=module.zero_mask.device,
                                          dtype=torch.bool)
                    off_mask.view(-1)[on_indices] = 0
                    module.off_mask = off_mask
                    module.rezero_weights()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("PruneLowMagnitude: Initialize")
        eo["pre_epoch"].append("PruneLowMagnitude: Maybe prune")
        return eo
