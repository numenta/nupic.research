# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from nupic.research.frameworks.vernon import (
    SelfSupervisedExperiment as SelfSupervisedExperimentBase,
)
from nupic.research.frameworks.vernon.distributed import (
    SupervisedExperiment as DistributedSupervisedExperiment,
)

__all__ = ["SelfSupervisedExperiment"]


class SelfSupervisedExperiment(
    DistributedSupervisedExperiment, SelfSupervisedExperimentBase
):
    """
    Distributed SelfSupervisedExperiment.
    """

    @classmethod
    def create_unsupervised_sampler(cls, config, dataset):
        return cls.create_train_sampler(config, dataset)

    @classmethod
    def create_supervised_sampler(cls, config, dataset):
        return cls.create_train_sampler(config, dataset)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "DistributedSelfSupervisedExperiment"

        eo.update(
            # Overwritten methods
            create_unsupervised_sampler=[exp + ": Create distributed sampler"],
            create_supervised_sampler=[exp + ": Create distributed sampler"],
        )
        return eo
