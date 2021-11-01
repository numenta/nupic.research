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

import copy

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from nupic.research.frameworks.pytorch.distributed_sampler import (
    UnpaddedDistributedSampler,
)
from nupic.research.frameworks.pytorch.model_utils import aggregate_eval_results
from nupic.research.frameworks.vernon.distributed.experiments.components import (
    DistributedBase,
)
from nupic.research.frameworks.vernon.experiments.supervised_experiment import (
    SupervisedExperiment as SupervisedExperimentBase,
)

__all__ = ["SupervisedExperiment"]


class SupervisedExperiment(DistributedBase, SupervisedExperimentBase):
    """
    Distributed SupervisedExperiment.

    TODO: Document find_unused_parameters for reference
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)
        if self.distributed:
            self.model = DistributedDataParallel(
                self.model,
                find_unused_parameters=config.get("find_unused_parameters", False),
            )
        else:
            self.model = DataParallel(self.model)

    def pre_epoch(self):
        super().pre_epoch()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

    @classmethod
    def create_train_sampler(cls, config, dataset):
        if config.get("distributed", False):
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
        return sampler

    @classmethod
    def create_validation_sampler(cls, config, dataset):
        if config.get("distributed", False):
            sampler = UnpaddedDistributedSampler(dataset, shuffle=False)
        else:
            sampler = None
        return sampler

    @classmethod
    def aggregate_results(cls, results):
        return cls._aggregate_validation_results(results)

    @classmethod
    def _aggregate_validation_results(cls, results):
        result = copy.copy(results[0])
        result.update(aggregate_eval_results(results))
        return result

    @classmethod
    def aggregate_pre_experiment_results(cls, results):
        if results[0] is not None:
            return cls._aggregate_validation_results(results)

        return None

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "DistributedSupervisedExperiment"

        # Extended methods
        eo["setup_experiment"].append(exp + ": DistributedDataParallel")
        eo["pre_epoch"].append(exp + ": Update distributed sampler")

        eo.update(
            # Overwritten methods
            create_train_sampler=[exp + ": Create distributed sampler"],
            create_validation_sampler=[exp + ": Create distributed sampler"],
            aggregate_results=[exp + ": Aggregate validation results"],
            aggregate_pre_experiment_results=[exp + ": Aggregate validation results"],
        )

        return eo
