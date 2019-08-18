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

from torch import nn

from nupic.torch.modules import Flatten


class SimpleCNN(nn.Sequential):
    """Simple CNN model for testing batch norm removal. One CNN layer plus
    one fully connected layer plus a linear output layer

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    """

    def __init__(self,
                 cnn_out_channels=(2, 2),
                 linear_units=3,
                 ):
        super(SimpleCNN, self).__init__()
        # input_shape = (1, 32, 32)
        # First Sparse CNN layer
        self.add_module("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5))
        self.add_module("cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0],
                                                         affine=False))
        self.add_module("cnn1_maxpool", nn.MaxPool2d(2))
        self.add_module("cnn1_relu", nn.ReLU())

        self.add_module("flatten", Flatten())

        # Sparse Linear layer
        self.add_module("linear", nn.Linear(196 * cnn_out_channels[0], linear_units))
        self.add_module("linear_bn", nn.BatchNorm1d(linear_units, affine=False))
        self.add_module("linear_relu", nn.ReLU())

        # Classifier
        self.add_module("output", nn.Linear(linear_units, 12))
