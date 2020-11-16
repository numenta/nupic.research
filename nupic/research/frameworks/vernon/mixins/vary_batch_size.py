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

from collections.abc import Sequence


class VaryBatchSize(object):
    """
    This mixin enables loading the training data with varying batch-sizes from
    epoch to epoch.

    :param config:
        - batch_sizes: list of batch sizes; the last one will be used for the remainder
                       of training once all have been exhausted, going in order
    """

    def setup_experiment(self, config):
        # Get and validate batch sizes.
        batch_sizes = config.get("batch_sizes", None)
        assert isinstance(batch_sizes, Sequence), "Must specify list of batch sizes"
        super().setup_experiment(
            {**config, "batch_size": batch_sizes[0]}
        )

        self.batch_sizes = batch_sizes

    @classmethod
    def _create_train_dataloader(cls, config, dataset, sampler, epoch):
        # Set the train dataloader to the appropriate batch size.
        batch_sizes = config["batch_sizes"]
        batch_size = batch_sizes[min(epoch, len(batch_sizes) - 1)]
        return super()._create_train_dataloader(
            {**config, "batch_size": batch_size},
            dataset, sampler, epoch
        )

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch < len(self.batch_sizes):
            self.logger.info("Setting batch_size=%s (variant %s)",
                             self.train_loader.batch_size,
                             self.current_epoch)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "VaryBatchSize: "
        eo["setup_experiment"].insert(0, name + "set initial batch size")
        eo["setup_experiment"].append(name + "initialization")
        eo["_create_train_dataloader"].append(
            name + "set the batch-size")
        eo["pre_epoch"].append(
            name + "log modified batch size")
        return eo
