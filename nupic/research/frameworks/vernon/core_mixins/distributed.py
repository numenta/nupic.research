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

import torch.distributed as dist
from torch import multiprocessing

__all__ = [
    "Distributed",
]


class Distributed:
    """
    Extends the BaseExperiment to work in distributed scenarios.

    Any class inheriting from Distributed should:
    - Implement aggregate_results and aggregate_pre_experiment_results
      if results from different processes need to be aggregated.
    - Wrap models with DistributedDataParallel
    - Use DistributedSamplers when using dataloaders
    - Update samplers with `sampler.set_epoch()`
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - distributed: Whether or not to use Pytorch Distributed training
            - backend: Pytorch Distributed backend ("nccl", "gloo")
                    Default: nccl
            - world_size: Total number of processes participating
            - rank: Rank of the current process
        """
        distributed = config.get("distributed", False)
        rank = config.get("rank", 0)

        if distributed and rank != 0:
            # Only enable logs from first process
            config = copy.copy(config)
            config["disable_logger"] = True

        super().setup_experiment(config)

        self.distributed = distributed
        self.rank = rank

        # CUDA runtime does not support the fork start method.
        # See https://pytorch.org/docs/stable/notes/multiprocessing.html
        multiprocessing.set_start_method("spawn", force=True)

        if self.distributed:
            dist_url = config.get("dist_url", "tcp://127.0.0.1:54321")
            backend = config.get("backend", "nccl")
            world_size = config.get("world_size", 1)
            dist.init_process_group(
                backend=backend,
                init_method=dist_url,
                rank=self.rank,
                world_size=world_size,
            )

    def stop_experiment(self):
        super().stop_experiment()
        if self.distributed:
            dist.destroy_process_group()

    @classmethod
    def aggregate_results(cls, results):
        """
        Aggregate multiple processes' "run_iteration" results into a single
        result. The default implementation does no aggregation and simply
        returns the first result.

        :param results:
            A list of return values from run_iteration from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """
        return results[0]

    @classmethod
    def aggregate_pre_experiment_results(cls, results):
        """
        Aggregate multiple processes' "run_pre_experiment" results into a single
        result. The default implementation does no aggregation and simply
        returns the first result.

        :param results:
            A list of return values from run_iteration from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """
        return results[0]

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "Distributed"

        # Extended methods
        eo["setup_experiment"].insert(0, name + ": Check if logging enabled")
        eo["setup_experiment"].append(name + ": Initialize")
        eo["stop_experiment"].append(name + ": Destroy processes")

        eo.update(
            # New methods
            aggregate_results=[name + ": Skipping aggregation"],
            aggregate_pre_experiment_results=[name + ": Skipping aggregation"],
        )

        return eo
