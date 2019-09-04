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

from dynamic_sparse.networks.layers import DSConv2d, SparseConv2d, calc_sparsity

from .main import BaseModel


class DSCNN(BaseModel):
    """
    Similar to other sparse models, but the focus here is on convolutional layers as
    opposed to dense layers.
    """

    log_attrs = [
        "pruning_iterations",
        "kept_frac",
        "prune_mask_sparsity",
        "keep_mask_sparsity",
        "weight_sparsity",
        "last_coactivations",
    ]

    def _post_epoch_updates(self, dataset=None):

        super()._post_epoch_updates(dataset)

        for name, module in self.network.named_modules():

            if isinstance(module, DSConv2d):
                # Log coactivation before pruning - otherwise they get reset.
                self.log["hist_" + "coactivations_" + name] = module.coactivations
                # Prune. Then log some params.
                module.progress_connections()
                for attr in self.log_attrs:
                    value = getattr(module, attr) if hasattr(module, attr) else -2
                    if isinstance(value, Iterable):
                        attr = "hist_" + attr
                    self.log[attr + "_" + name] = value

            if isinstance(module, (DSConv2d, SparseConv2d)):
                self.log["sparsity_" + name] = calc_sparsity(module.weight)

    def _log_weights(self):
        """Log weights for all layers which have params."""
        if "param_layers" not in self.__dict__:
            self.param_layers = defaultdict(list)
            for m, ltype in [(m, self.has_params(m)) for m in self.network.modules()]:
                if ltype:
                    self.param_layers[ltype].append(m)
