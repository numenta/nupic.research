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

from nupic.research.frameworks.pytorch.dataset_utils import (
    FunctionalPreprocessedDataset,
)


class LoadPreprocessedData(object):
    """
    This mixin loads the appropriate preprocessed dataset for each epoch.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)
        assert isinstance(self.train_dataset, FunctionalPreprocessedDataset), (
            "The train dataset is not preprocessed; there's nothing to load."
        )

    @classmethod
    def _create_train_dataloader(cls, config, dataset, sampler, epoch):
        dataset = dataset.get_variant(epoch % dataset.num_variants())
        return super()._create_train_dataloader(config, dataset, sampler, epoch)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "LoadPreprocessedData"
        eo["setup_experiment"].append(name + ": ensure the train set is preprocessed")
        eo["_create_train_dataloader"].append(
            name + ": load the specified epoch's data")
        return eo
