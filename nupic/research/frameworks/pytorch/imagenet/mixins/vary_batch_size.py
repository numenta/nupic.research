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

from copy import deepcopy


class VaryBatchSize(object):
    """
    This mixin enables loading the training data with varying batch-sizes from
    epoch to epoch.

    :param config:
        - batch_sizes: list of batch sizes; the last one will be used for the remainder
                       of training once all have been exhausted, going in order
        - batch_size: Can be given as a trivial case where the batch size doesn't vary;
                      can only be used when `batch_sizes=None`
    """

    def setup_experiment(self, config):

        # Validate batch sizes.
        batch_sizes = config.get("batch_sizes", None)
        batch_size = config.get("batch_size", None)

        # Assert exclusive or; only one can't be None.
        assert (batch_size is not None) != (batch_sizes is not None), (
            "Must specify one of 'batch_size' or 'batch_sizes', not both.")

        # Set the default list of batch sizes where the won't vary epoch to epoch.
        if batch_size is not None:
            batch_sizes = [batch_size]
        # Ensure batch_sizes is a list.
        if batch_sizes is not None:
            assert isinstance(batch_sizes, list), "Must specify list of batch sizes"

        self.batch_sizes = batch_sizes
        super().setup_experiment(config)

        # super() will set up a loader for the first batch size. Now the remaining..
        train_set = self.train_loader.dataset

        self.train_loaders = [self.train_loader]
        for i in range(1, len(batch_sizes)):
            self.train_loaders.append(self.create_train_dataloader(
                train_set, config, batch_size_variant=i
            ))

        self.current_batch_size_variant = 0
        self.batch_size = self.train_loader.batch_size

    def post_epoch(self):
        """
        Set the next train dataloader to set the next batch-size.
        """
        super().post_epoch()
        i = self.current_batch_size_variant
        self.current_batch_size_variant = min(i + 1, len(self.train_loaders) - 1)
        self.train_loader = self.train_loaders[self.current_batch_size_variant]
        self.logger.info("Setting batch_size={} (variant {})".format(
            self.train_loader.batch_size, self.current_batch_size_variant
        ))

    @classmethod
    def create_train_dataloader(cls, dataset, config, batch_size_variant=0):
        """
        Create a train dataloader based off `batch_size_variant` if multiple
        batch_sizes are specified.
        """
        batch_sizes = config.get("batch_sizes", None)
        if batch_sizes is not None:
            assert batch_size_variant < len(batch_sizes)
            batch_size = batch_sizes[batch_size_variant]
            config = deepcopy(config)
            config.update(batch_size=batch_size)
        else:
            assert batch_size_variant == 0, "There's only one batch size."

        return super().create_train_dataloader(dataset, config)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "VaryBatchSize: "
        eo["setup_experiment"].append(name + "load one train dataloader per batch-size")
        eo["post_epoch"].append(name + "set the next dataloader\\batch-size")
        return eo
