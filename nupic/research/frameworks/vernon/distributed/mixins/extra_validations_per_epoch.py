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

from nupic.research.frameworks.pytorch.model_utils import aggregate_eval_results
from nupic.research.frameworks.vernon import interfaces
from nupic.research.frameworks.vernon.mixins.extra_validations_per_epoch import (
    ExtraValidationsPerEpoch as ExtraValidationsBase,
)

__all__ = [
    "ExtraValidationsPerEpoch",
]


class ExtraValidationsPerEpoch(interfaces.DistributedAggregation,  # Requires
                               ExtraValidationsBase):
    @classmethod
    def aggregate_results(cls, results):
        ret = super().aggregate_results(results)

        extra_val_aggregated = []
        for i in range(len(ret["extra_val_results"])):
            timestep = ret["extra_val_results"][i][0]
            val_results = [process_result["extra_val_results"][i][1]
                           for process_result in results]
            extra_val_aggregated.append(
                (timestep, aggregate_eval_results(val_results))
            )
        ret["extra_val_results"] = extra_val_aggregated

        return ret

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        # Extended methods
        eo["aggregate_results"].append(
            "ExtraValidationsDistributed: Aggregate extra validations")
        return eo
