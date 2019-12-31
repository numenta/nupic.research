# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from collections import OrderedDict
from functools import partial

from torch import nn

from nupic.research.frameworks.pytorch.modules import KWinners2dLocal
from nupic.torch.modules import Flatten, KWinners, KWinners2d

__all__ = [
    "AlexNetKWinners",
    "gsc_alexnet_kwinners",
]


class AlexNetKWinners(nn.Sequential):

    def __init__(self,
                 input_size,
                 num_classes,
                 cnn_out_channels=(64, 64),
                 cnn_activity_percent_on=(0.095, 0.125),
                 linear_units=1000,
                 linear_activity_percent_on=(0.1,),
                 kernel_size=5,
                 maxpool_stride=2,
                 boost_strength=1.5,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.0,
                 use_kwinners_local=False):
        feature_map_sidelength = (
            (((input_size[1] - kernel_size + 1) / maxpool_stride)
             - kernel_size + 1) / maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        kwinner2d_class = KWinners2dLocal if use_kwinners_local else KWinners2d

        super().__init__(OrderedDict([

            # -------------
            # Conv Block
            # -------------

            ("cnn1", nn.Conv2d(input_size[0],
                               cnn_out_channels[0],
                               kernel_size)),
            ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0],
                                       affine=False)),
            ("cnn1_kwinner", kwinner2d_class(
                channels=cnn_out_channels[0],
                percent_on=cnn_activity_percent_on[0],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),

            # -------------
            # Conv Block
            # -------------

            ("cnn2", nn.Conv2d(cnn_out_channels[0],
                               cnn_out_channels[1],
                               kernel_size)),
            ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[1],
                                       affine=False)),
            ("cnn2_kwinner", kwinner2d_class(
                channels=cnn_out_channels[1],
                percent_on=cnn_activity_percent_on[1],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("flatten", Flatten()),

            # -------------
            # Linear Block
            # -------------

            ("fc1", nn.Linear(
                (feature_map_sidelength**2) * cnn_out_channels[1],
                linear_units)),
            ("fc1_bn", nn.BatchNorm1d(linear_units, affine=False)),
            ("fc1_kwinner", KWinners(
                n=linear_units,
                percent_on=linear_activity_percent_on[0],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("fc1_dropout", nn.Dropout(0.5)),

            # -------------
            # Output Layer
            # -------------

            ("fc2", nn.Linear(linear_units,
                              num_classes)),

        ]))


gsc_alexnet_kwinners = partial(AlexNetKWinners,
                               input_size=(1, 32, 32),
                               num_classes=12,
                               cnn_activity_percent_on=(0.095, 0.125),
                               linear_activity_percent_on=(0.1,),
                               boost_strength=1.5,
                               boost_strength_factor=0.9,
                               duty_cycle_period=1000,
                               k_inference_factor=1.0)
