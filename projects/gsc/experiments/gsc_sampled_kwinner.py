#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Run a GSC experiment using SparseLeNet with SampledKWinners.
"""

from copy import deepcopy

from .sampled_kwinners.sampled_le_sparse_net import SampledLeSparseNet
from .sampled_kwinners.update_kwinner_temperature import UpdateKWinnerTemperature
from .base import DEFAULT_SPARSE_CNN

from nupic.research.frameworks.vernon import experiments, mixins


class SampledKWinnersGSCExperiment(mixins.VaryBatchSize, mixins.RezeroWeights, UpdateKWinnerTemperature,
                                   mixins.LoadPreprocessedData,
                                   experiments.SupervisedExperiment):
    pass

SPARSE_CNN_SAMPLED_KWINNER = deepcopy(DEFAULT_SPARSE_CNN)
SPARSE_CNN_SAMPLED_KWINNER.update(
    experiment_class=SampledKWinnersGSCExperiment,
    model_class=SampledLeSparseNet,
    model_args=dict(
        input_shape=(1, 32, 32),
        cnn_out_channels=(64, 64),
        cnn_activity_percent_on=(0.095, 0.125),
        cnn_weight_percent_on=(0.5, 0.2),
        linear_n=(1000,),
        linear_activity_percent_on=(0.1,),
        linear_weight_percent_on=(0.1,),
        temperature=10.0,
        eval_temperature=1.0,
        temperature_decay_rate=0.01,
        use_batch_norm=True,
        dropout=0.0,
        num_classes=12,
        k_inference_factor=1.0,
        activation_fct_before_max_pool=True,
        consolidated_sparse_weights=False,
    ),
    epochs_to_validate=range(30),
)


CONFIGS = dict(
    sparse_cnn_sampled_kwinner=SPARSE_CNN_SAMPLED_KWINNER,
)
