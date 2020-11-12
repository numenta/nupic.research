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

from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from nupic.research.frameworks.pytorch.dataset_utils.samplers import (
    TaskDistributedSampler,
)
from nupic.research.frameworks.vernon.distributed.experiments.components import (
    DistributedBase,
)
from nupic.research.frameworks.vernon.experiments.cl_experiment import (
    ContinualLearningExperiment as ContinualLearningExperimentBase,
)

__all__ = [
    "ContinualLearningExperiment",
]


class ContinualLearningExperiment(DistributedBase,
                                  ContinualLearningExperimentBase):
    """
    ContinualLearningExperiment for distributed experiments. Distributed
    validation is not implemented.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        if self.distributed:
            self.model = DistributedDataParallel(self.model)
        else:
            self.model = DataParallel(self.model)

    def pre_epoch(self):
        super().pre_epoch()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

    @classmethod
    def create_train_sampler(cls, config, dataset):
        if config.get("distributed", False):
            task_indices = cls.compute_task_indices(config, dataset)
            return TaskDistributedSampler(
                dataset,
                task_indices
            )
        else:
            return super().create_train_sampler(config, dataset)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "DistributedContinualLearningExperiment"

        # Extended methods
        eo["setup_experiment"].append(exp + ": DistributedDataParallel")
        eo["pre_epoch"].append(exp + ": Update distributed sampler")
        eo["create_train_sampler"].insert(0,
                                          ("If distributed { "
                                           "create distribited sampler "
                                           "} else {"))
        eo["create_train_sampler"].append("}")
        # FIXME: Validation is not currently distributed. Implement
        # create_validation_sampler and aggregate_results.

        return eo
