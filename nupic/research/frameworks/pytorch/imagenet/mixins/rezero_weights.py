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
from nupic.torch.modules import rezero_weights


class RezeroWeights:
    """
    Rezero the SparseWeights after every epoch.

    Note that the SparseWeights rezeroes weights during the forward pass during
    learning, so this mixin only needs to be applied on post_epoch rather than
    post_batch.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)

        if self.rank == 0:
            params_sparse, nonzero_params_sparse2 = count_nonzero_params(
                self.model)
            self.logger.debug("Params total/nnz %s / %s = %s ",
                              params_sparse, nonzero_params_sparse2,
                              float(nonzero_params_sparse2) / params_sparse)

    @classmethod
    def create_model(cls, config, device):
        model = super().create_model(config, device)
        # Some initialization strategies can destroy sparsity, so we call rezero
        # here.
        model.apply(rezero_weights)
        return model

    def post_batch(self, model, error_loss, complexity_loss, batch_idx,
                   *args, **kwargs):
        super().post_batch(model, error_loss, complexity_loss, batch_idx,
                           *args, **kwargs)

        extra_validate = (batch_idx in self.additional_batches_to_validate
                          and self.current_epoch in self.epochs_to_validate)
        if extra_validate:
            self.model.apply(rezero_weights)

    def post_epoch(self):
        super().post_epoch()

        count_nnz = self.logger.isEnabledFor(logging.DEBUG) and self.rank == 0
        if count_nnz:
            params_sparse, nonzero_params_sparse1 = count_nonzero_params(
                self.model)

        self.model.apply(rezero_weights)

        if count_nnz:
            params_sparse, nonzero_params_sparse2 = count_nonzero_params(
                self.model)
            self.logger.debug(
                "Params total/nnz before/nnz after %s %s / %s = %s",
                params_sparse, nonzero_params_sparse1,
                nonzero_params_sparse2,
                float(nonzero_params_sparse2) / params_sparse)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("RezeroWeights logging")
        eo["create_model"].append("RezeroWeights")
        eo["post_batch"].append("RezeroWeights if about to validate")
        eo["post_epoch"].append("RezeroWeights")
        return eo
