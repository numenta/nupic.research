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


import torch
from .block_model_experiment import BlockModelExperiment
from torch.nn.parallel import DataParallel

class DataParallelBlockModelExperiment(BlockModelExperiment):
    def setup_experiment(self, config):
        super(DataParallelBlockModelExperiment, self).setup_experiment(config)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        self.encoder_classifier = DataParallel(self.encoder_classifier)
        self.encoder = DataParallel(self.encoder)