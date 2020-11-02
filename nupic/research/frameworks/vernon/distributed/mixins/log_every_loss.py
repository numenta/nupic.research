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

import torch

from nupic.research.frameworks.vernon import interfaces
from nupic.research.frameworks.vernon.mixins.log_every_loss import (
    LogEveryLoss as LogEveryLossBase,
)

__all__ = [
    "LogEveryLoss",
]


class LogEveryLoss(interfaces.DistributedAggregation,  # Requires
                   LogEveryLossBase):
    @classmethod
    def aggregate_results(cls, results):
        aggregated = super().aggregate_results(results)

        k = "error_loss_history"
        if k in aggregated:
            loss_by_process_and_batch = torch.Tensor(len(results),
                                                     len(results[0][k]))
            for rank, result in enumerate(results):
                loss_by_process_and_batch[rank, :] = torch.tensor(result[k])
            aggregated[k] = loss_by_process_and_batch.mean(dim=0).tolist()

        # "complexity_loss_history" doesn't need to be aggregated, since it's
        # the same on every process.

        return aggregated

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["aggregate_results"].append("LogEveryLoss: Aggregate")
        return eo
