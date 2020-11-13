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

        self.config = config
        self.batch_sizes = batch_sizes

    def prepare_loaders_for_epoch(self, epoch):
        """
        Set the train dataloader to the appropriate batch size.
        """
        super().prepare_loaders_for_epoch(epoch)
        batch_size = self.batch_sizes[min(epoch,
                                          len(self.batch_sizes) - 1)]
        if self.train_loader.batch_size != batch_size:
            self.train_loader = self.create_train_dataloader(
                {**self.config, "batch_size": batch_size}
            )

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch < len(self.batch_sizes):
            # Log this here, not in prepare_loaders_for_epoch, because
            # prepare_loaders_for_epoch is also called in compute_steps_in_epoch.
            self.logger.info("Setting batch_size=%s (variant %s)",
                             self.train_loader.batch_size,
                             self.current_epoch)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "VaryBatchSize: "
        eo["setup_experiment"].insert(0, name + "set initial batch size")
        eo["setup_experiment"].append(name + "initialization")
        eo["prepare_loaders_for_epoch"].append(
            name + "set the dataloader and batch-size")
        eo["pre_epoch"].append(
            name + "log modified batch size")
        return eo
