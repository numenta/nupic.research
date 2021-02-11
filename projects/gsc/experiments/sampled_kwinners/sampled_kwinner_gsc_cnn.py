#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import warnings

from torch import nn

from nupic.torch.modules import Flatten, SparseWeights, SparseWeights2d

from .sampled_kwinners import SampledKWinners, SampledKWinners2d


class SampledKWinnerGSCSparseCNN(nn.Sequential):
    """Sparse CNN model used to classify `Google Speech Commands` dataset as
    described in `How Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    :param kwinner_local: Whether or not to choose the k-winners locally (across
                          the channels at each location) or globally (across the
                          whole input and across all channels)
    :param cnn_sparsity: Percent of weights that zero
    :param linear_sparsity: Percent of weights that are zero in the
                            linear layer.
    """

    def __init__(
        self,
        cnn_out_channels=(64, 64),
        cnn_percent_on=(0.095, 0.125),
        cnn_weight_sparsity=None,
        linear_units=1000,
        linear_percent_on=0.1,
        linear_weight_sparsity=None,
        temperature=10.0,
        eval_temperature=1.0,
        temperature_decay_rate=0.01,
        k_inference_factor=1.0,
        cnn_sparsity=(0.5, 0.8),
        linear_sparsity=0.9,
    ):
        super(SampledKWinnerGSCSparseCNN, self).__init__()

        if cnn_weight_sparsity is not None:
            warnings.warn(
                "Parameter `cnn_weight_sparsity` is deprecated. Use "
                "`cnn_sparsity` instead.",
                DeprecationWarning,
            )
            cnn_sparsity = (1.0 - cnn_weight_sparsity[0], 1.0 - cnn_weight_sparsity[1])

        if linear_weight_sparsity is not None:
            warnings.warn(
                "Parameter `linear_weight_sparsity` is deprecated. Use "
                "`linear_sparsity` instead.",
                DeprecationWarning,
            )
            linear_sparsity = 1.0 - linear_weight_sparsity

        # input_shape = (1, 32, 32)
        # First Sparse CNN layer
        if cnn_sparsity[0] > 0:
            self.add_module(
                "cnn1",
                SparseWeights2d(
                    nn.Conv2d(1, cnn_out_channels[0], 5), sparsity=cnn_sparsity[0]
                ),
            )
        else:
            self.add_module("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5))
        self.add_module(
            "cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0], affine=False)
        )
        self.add_module(
            "cnn1_kwinner",
            SampledKWinners2d(
                percent_on=cnn_percent_on[0],
                k_inference_factor=k_inference_factor,
                temperature=temperature,
                eval_temperature=eval_temperature,
                temperature_decay_rate=temperature_decay_rate,
                relu=False,
            ),
        )
        self.add_module("cnn1_maxpool", nn.MaxPool2d(2))

        # Second Sparse CNN layer
        if cnn_sparsity[1] > 0:
            self.add_module(
                "cnn2",
                SparseWeights2d(
                    nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5),
                    sparsity=cnn_sparsity[1],
                ),
            )
        else:
            self.add_module(
                "cnn2", nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5)
            )
        self.add_module(
            "cnn2_batchnorm", nn.BatchNorm2d(cnn_out_channels[1], affine=False)
        )
        self.add_module(
            "cnn2_kwinner",
            SampledKWinners2d(
                percent_on=cnn_percent_on[0],
                k_inference_factor=k_inference_factor,
                temperature=temperature,
                eval_temperature=eval_temperature,
                temperature_decay_rate=temperature_decay_rate,
                relu=False,
            ),
        )
        self.add_module("cnn2_maxpool", nn.MaxPool2d(2))

        self.add_module("flatten", Flatten())

        # Sparse Linear layer
        self.add_module(
            "linear",
            SparseWeights(
                nn.Linear(25 * cnn_out_channels[1], linear_units),
                sparsity=linear_sparsity,
            ),
        )
        self.add_module("linear_bn", nn.BatchNorm1d(linear_units, affine=False))
        self.add_module(
            "linear_kwinner",
            SampledKWinners(
                percent_on=linear_percent_on,
                k_inference_factor=k_inference_factor,
                temperature=temperature,
                eval_temperature=eval_temperature,
                temperature_decay_rate=temperature_decay_rate,
                relu=False,
            ),
        )

        # Classifier
        self.add_module("output", nn.Linear(linear_units, 12))
        self.add_module("softmax", nn.LogSoftmax(dim=1))
