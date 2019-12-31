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

import torch.nn.functional as F
from torch import nn

from nupic.research.frameworks.pytorch.modules import KWinners2dLocal
from nupic.torch.modules import Flatten, KWinners, KWinners2d

__all__ = [
    "AlexNetBinaryActivationKWinners",
    "gsc_alexnet_binaryactivation_kwinners",
]


class SetNonzerosToOne(nn.Module):
    def __init__(self, cancel_gradient, inplace):
        super().__init__()
        self.cancel_gradient = cancel_gradient
        self.inplace = inplace

    def forward(self, x):
        if self.cancel_gradient:
            x = F.hardtanh(x, -1, 1)
        if not self.inplace:
            x = x.clone()
        mask = x.data != 0
        x.data[mask] = 1
        x.data[~mask] = 0
        return x


    class AlexNetBinaryActivationKWinners(nn.Sequential):

    def __init__(self, input_size, num_classes,
                 boost_strength=1.5,
                 boost_strength_factor=0.9,
                 duty_cycle_period=1000,
                 k_inference_factor=1.0,
                 use_kwinners_local=False,
                 bn_track_running_stats=True):
        ratio_infl = 3

        nonlin_params = dict(cancel_gradient=False,
                             inplace=False)

        kwinner2d_class = KWinners2dLocal if use_kwinners_local else KWinners2d

        super().__init__(OrderedDict([

            ("cnn1", nn.Conv2d(
                input_size[0], 64 * ratio_infl, kernel_size=5, stride=2,
                padding=1)),
            ("cnn1_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("cnn1_bn", nn.BatchNorm2d(
                64 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn1_kwinner", kwinner2d_class(
                channels=64 * ratio_infl,
                percent_on=0.095,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("cnn1_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("cnn2", nn.Conv2d(
                64 * ratio_infl, 192 * ratio_infl, kernel_size=5, padding=2)),
            ("cnn2_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("cnn2_bn", nn.BatchNorm2d(
                192 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn2_kwinner", kwinner2d_class(
                channels=192 * ratio_infl,
                percent_on=0.125,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("cnn2_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("cnn3", nn.Conv2d(
                192 * ratio_infl, 384 * ratio_infl, kernel_size=3, padding=1)),
            ("cnn3_bn", nn.BatchNorm2d(
                384 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn3_kwinner", kwinner2d_class(
                channels=384 * ratio_infl,
                percent_on=0.15,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("cnn3_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("cnn4", nn.Conv2d(
                384 * ratio_infl, 256 * ratio_infl, kernel_size=3, padding=1)),
            ("cnn4_bn", nn.BatchNorm2d(
                256 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn4_kwinner", kwinner2d_class(
                channels=256 * ratio_infl,
                percent_on=0.15,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("cnn4_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("cnn5", nn.Conv2d(
                256 * ratio_infl, 256, kernel_size=3, padding=1)),
            ("cnn5_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("cnn5_bn", nn.BatchNorm2d(
                256,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn5_kwinner", kwinner2d_class(
                channels=256,
                percent_on=0.15,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("cnn5_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("flatten", Flatten()),

            ("fc1", nn.Linear(256, 4096)),
            ("fc1_bn", nn.BatchNorm1d(
                4096,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("fc1_kwinner", KWinners(
                n=4096,
                percent_on=0.1,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("fc1_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("fc2", nn.Linear(4096, 4096)),
            ("fc2_bn", nn.BatchNorm1d(
                4096,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("fc1_kwinner", KWinners(
                n=4096,
                percent_on=0.1,
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            )),
            ("fc2_nonlin", SetNonzerosToOne(**nonlin_params)),

            ("fc3", nn.Linear(4096, num_classes)),
            ("fc3_bn", nn.BatchNorm1d(
                num_classes,
                affine=False,
                track_running_stats=bn_track_running_stats)),
        ]))


gsc_alexnet_binaryactivation_kwinners = partial(AlexNetBinaryActivationKWinners,
                                                input_size=(1, 32, 32),
                                                num_classes=12)
