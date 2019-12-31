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

from nupic.torch.modules import Flatten

__all__ = [
    "AlexNet",
    "gsc_alexnet",
]


class AlexNet(nn.Sequential):

    def __init__(self, input_size, num_classes,
                 bn_track_running_stats=True):
        ratio_infl = 3

        super().__init__(OrderedDict([

            ("cnn1", nn.Conv2d(
                input_size[0], 64 * ratio_infl, kernel_size=5, stride=2,
                padding=1)),
            ("cnn1_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("cnn1_bn", nn.BatchNorm2d(
                64 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn1_nonlin", nn.ReLU()),

            ("cnn2", nn.Conv2d(
                64 * ratio_infl, 192 * ratio_infl, kernel_size=5, padding=2)),
            ("cnn2_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("cnn2_bn", nn.BatchNorm2d(
                192 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn2_nonlin", nn.ReLU()),

            ("cnn3", nn.Conv2d(
                192 * ratio_infl, 384 * ratio_infl, kernel_size=3, padding=1)),
            ("cnn3_bn", nn.BatchNorm2d(
                384 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn3_nonlin", nn.ReLU()),

            ("cnn4", nn.Conv2d(
                384 * ratio_infl, 256 * ratio_infl, kernel_size=3, padding=1)),
            ("cnn4_bn", nn.BatchNorm2d(
                256 * ratio_infl,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn4_nonlin", nn.ReLU()),

            ("cnn5", nn.Conv2d(
                256 * ratio_infl, 256, kernel_size=3, padding=1)),
            ("cnn5_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
            ("cnn5_bn", nn.BatchNorm2d(
                256,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("cnn5_nonlin", nn.ReLU()),

            ("flatten", Flatten()),

            ("fc1", nn.Linear(256, 4096)),
            ("fc1_bn", nn.BatchNorm1d(
                4096,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("fc1_nonlin", nn.ReLU()),
            ("fc1_dropout", nn.Dropout(0.5)),

            ("fc2", nn.Linear(4096, 4096)),
            ("fc2_bn", nn.BatchNorm1d(
                4096,
                affine=False,
                track_running_stats=bn_track_running_stats)),
            ("fc2_nonlin", nn.ReLU()),
            ("fc2_dropout", nn.Dropout(0.5)),

            ("fc3", nn.Linear(4096, num_classes)),
            ("fc3_bn", nn.BatchNorm1d(
                num_classes,
                affine=False,
                track_running_stats=bn_track_running_stats)),
        ]))


gsc_alexnet = partial(AlexNet, input_size=(1, 32, 32), num_classes=12)
