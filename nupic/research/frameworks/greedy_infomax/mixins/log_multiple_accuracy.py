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
from nupic.research.frameworks.greedy_infomax.utils.train_utils import evaluate_block_model
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import multiple_cross_entropy

__all__ = [
    "LogMultipleAccuracy",
]


class LogMultipleAccuracy:
    """
    At the end of each epoch, log the individual accuracies of each classification head
    (which correspond to an EmitEncoding module).
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.evaluate_model_func = evaluate_block_model
        self._loss_function_supervised = multiple_cross_entropy