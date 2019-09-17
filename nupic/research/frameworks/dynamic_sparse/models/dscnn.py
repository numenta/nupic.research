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

import re
from collections import defaultdict
from collections.abc import Iterable

from nupic.research.frameworks.dynamic_sparse.networks.layers import (
    DSConv2d,
    SparseConv2d,
    calc_sparsity,
    init_coactivation_tracking,
)

from .main import BaseModel


class DSCNN(BaseModel):
    """
    Similar to other sparse models, but the focus here is on convolutional layers as
    opposed to dense layers.
    """

    log_attrs = [
        "pruning_iterations",
        "prune_mask_sparsity",
        "on2off_mask_sparsity",
        "on2off_mask_sparsity",
        "on2off_mask_num",
        "off2on_mask_num",
        "keep_mask_sparsity",
        "weight_sparsity",
        "last_coactivations",
        "weight_c01x_mean",
        "weight_c01x_std",
        "weight_c11x_mean",
        "weight_c11x_std",
        "weight_c01x_c11x_mean_diff",
        "input_means",
        "output_means",
        "c10_num",
        "c10_frac",
        "c10_frac_rel",
        "c01_num",
        "c01_frac",
        "c01_frac_rel",
        "c11_num",
        "c11_frac",
        "c11_frac_rel",
        "c00_num",
        "c00_frac",
        "c00_frac_rel",
        "c000_frac",
        "c001_frac",
        "c010_frac",
        "c011_frac",
        "c100_frac",
        "c101_frac",
        "c110_frac",
        "c111_frac",
        "c000_frac_rel",
        "c001_frac_rel",
        "c010_frac_rel",
        "c011_frac_rel",
        "c100_frac_rel",
        "c101_frac_rel",
        "c110_frac_rel",
        "c111_frac_rel",
        "survival_rate",
        "c10_num",
        "c10_frac",
        "c10_frac_rel",
        "c01_num",
        "c01_frac",
        "c01_frac_rel",
        "tot_grad_flow",
        "c00_grad_flow",
        "c00_grad_flow_centered",
        "c01_grad_flow",
        "c01_grad_flow_centered",
        "c10_grad_flow",
        "c10_grad_flow_centered",
        "c11_grad_flow",
        "c11_grad_flow_centered",
        "c01_c11_grad_flow_diff",
        "c01_c11_grad_flow_diff_centered",
    ]

    def setup(self):
        super().setup()
        self.network.apply(init_coactivation_tracking)

    def _post_epoch_updates(self, dataset=None):

        super()._post_epoch_updates(dataset)

        for name, module in self.network.named_modules():

            name = re.sub(r"squashed" + r"(\d+)", "", name)
            total_attr_d = defaultdict(lambda: 0)
            count_attr_d = defaultdict(lambda: 0)
            if isinstance(module, DSConv2d) and not isinstance(module, SparseConv2d):
                # Prune. Then log some params.
                module.progress_connections()
                print("progressing")
                for attr in self.log_attrs:
                    value = getattr(module, attr) if hasattr(module, attr) else None
                    if value is None:
                        continue
                    if isinstance(value, Iterable):
                        # Log attr as a histogram
                        attr = "hist_" + attr
                    elif hasattr(value, "__add__"):
                        # Collect cumulative stats for single numbers.
                        total_attr_d[attr] += value
                        count_attr_d[attr] += 1
                    self.log[attr + "_" + name] = value
                module._reset_logging_params()

            # # Log average stats across layers.
            # for attr, total in total_attr_d.items():
            #     count = count_attr_d[attr]
            #     self.log[attr + "_" + "total_mean"] = total / count

            if isinstance(module, (DSConv2d, SparseConv2d)):
                self.log["sparsity_" + name] = calc_sparsity(module.weight)

    def _log_weights(self):
        """Log weights for all layers which have params."""
        if "param_layers" not in self.__dict__:
            self.param_layers = defaultdict(list)
            for m, ltype in [(m, self.has_params(m)) for m in self.network.modules()]:
                if ltype:
                    self.param_layers[ltype].append(m)
