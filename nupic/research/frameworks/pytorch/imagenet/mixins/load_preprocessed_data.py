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

from nupic.research.frameworks.pytorch.dataset_utils import PreprocessedDataset


class LoadPreprocessedData(object):
    """
    This mixin helps manage preprocessed by ensuring the next set is loaded
    following every epoch in preparation for the next.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)
        assert isinstance(self.train_loader.dataset, PreprocessedDataset), (
            "The train dataset is not preprocessed; there's nothing to load."
        )

    def post_epoch(self):
        super().post_epoch()
        self.train_loader.dataset.load_next()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        name = "LoadPreprocessedData"
        eo["setup_experiment"].append(name + ": ensure the train set is preprocessed")
        eo["post_epoch"].append(name + ": load the next set of preprocessed data")
        return eo
