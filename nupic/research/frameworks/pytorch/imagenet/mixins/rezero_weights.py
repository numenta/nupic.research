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

import logging

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.torch.modules.sparse_weights import SparseWeightsBase


class RezeroWeights:
    """
    Rezero the SparseWeights after every batch.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)

        self._rezero_modules = [module
                                for module in self.model.modules()
                                if isinstance(module, SparseWeightsBase)]

        if self.rank == 0:
            params_sparse, nonzero_params_sparse2 = count_nonzero_params(
                self.model)
            self.logger.debug("Params nnz/total %s / %s = %s ",
                              nonzero_params_sparse2, params_sparse,
                              float(nonzero_params_sparse2) / params_sparse)

    @classmethod
    def create_model(cls, config):
        model = super().create_model(config)
        # Some initialization strategies can destroy sparsity, so we call rezero
        # here.
        for module in model.modules():
            if isinstance(module, SparseWeightsBase):
                module.rezero_weights()
        return model

    def post_batch(self, *args, **kwargs):
        super().post_batch(*args, **kwargs)
        for module in self._rezero_modules:
            module.rezero_weights()

    def post_epoch(self):
        super().post_epoch()

        if self.logger.isEnabledFor(logging.DEBUG) and self.rank == 0:
            params_sparse, nonzero_params_sparse = count_nonzero_params(
                self.model)
            self.logger.debug(
                "Params nnz/total %s / %s = %s",
                nonzero_params_sparse,
                params_sparse,
                float(nonzero_params_sparse) / params_sparse)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("RezeroWeights logging")
        eo["create_model"].append("RezeroWeights")
        eo["post_batch"].append("RezeroWeights")
        eo["post_epoch"].append("RezeroWeights logging")
        return eo
