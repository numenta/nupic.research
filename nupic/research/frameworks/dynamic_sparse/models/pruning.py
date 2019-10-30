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

from .loggers import DSNNLogger
from .main import SparseModel
from .modules import PrunableModule


class PruningModel(SparseModel):
    """Allows progressively pruning, building dense to sparse models"""

    def setup(self, config=None):
        super().setup(config)

        # add specific defaults
        new_defaults = dict(
            target_final_density=0.1,
            start_pruning_epoch=None,
            end_pruning_epoch=None,
            pruning_interval=1,
        )
        new_defaults = {k: v for k, v in new_defaults.items() if k not in self.__dict__}
        self.__dict__.update(new_defaults)

        # interval
        if self.end_pruning_epoch is None:
            self.end_pruning_epoch = self.epochs
        if self.start_pruning_epoch is None:
            self.start_pruning_epoch = 1
        interval = (
            self.end_pruning_epoch - self.start_pruning_epoch
        ) / self.pruning_interval

        # set target density for each sparse module
        self._make_attr_iterable(
            "target_final_density", counterpart=self.sparse_modules
        )
        for idx, (target_final_density) in enumerate(self.target_final_density):
            module = self.sparse_modules[idx]
            module.target_final_density = target_final_density
            module.target_density = 1.0
            module.decay_amount = (1.0 - target_final_density) / interval

        # share same logger as DSNN for now
        self.logger = DSNNLogger(self, config=self.config)

    def _sparse_module_type(self):
        return PrunableModule

    def _post_epoch_updates(self, dataset=None):
        super()._post_epoch_updates(dataset)
        if (
            self.current_epoch >= self.start_pruning_epoch
            and self.current_epoch <= self.end_pruning_epoch
            and self.current_epoch % self.pruning_interval == 0
        ):
            self.prune_network()

    def prune_network(self):
        # define how much pruning is done
        # print("Pruning in epoch {}".format(str(self.current_epoch)))
        for module in self.sparse_modules:
            module.prune()
            module.decay_density()
