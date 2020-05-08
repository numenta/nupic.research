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
    def setup_experiment(self, config):
        super().setup_experiment(config)
        if self.rank == 0:
            params_sparse, nonzero_params_sparse2 = count_nonzero_params(
                self.model)
            self.logger.debug("Params total/nnz %s / %s = %s ",
                              params_sparse, nonzero_params_sparse2,
                              float(nonzero_params_sparse2) / params_sparse)

    def post_epoch(self, epoch):
        super().post_epoch(epoch)

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
