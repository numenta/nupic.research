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

"""
Sparse Variational Dropout network
"""

from collections import OrderedDict
from functools import partial

from torch import nn

from nupic.research.frameworks.backprop_structure.modules.vdrop_layers import (
    VDropConv2d,
    VDropLinear,
    VDropCentralData
)


class VDropLenet(nn.Sequential):

    def __init__(self, input_shape=(1, 32, 32),
                 cnn_out_channels=(64, 64),
                 num_classes=12,
                 use_batch_norm=True,
                 z_logvar_init=-10,
                 vdrop_data_class=VDropCentralData,
                 kernel_size=5,
                 linear_units=1000,
                 maxpool_stride=2,
                 bn_track_running_stats=True
                 ):
        feature_map_sidelength = (
            (((input_size[1] - kernel_size + 1) / maxpool_stride)
             - kernel_size + 1) / maxpool_stride
        )
        vdrop_data = vdrop_data_class(z_logvar_init=z_logvar_init)

        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        modules = [
            # -------------
            # Conv Block
            # -------------

            ("vdrop_cnn1", VDropConv2d(input_size[0],
                                 cnn_out_channels[0],
                                 kernel_size, vdrop_data)),
            ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
        ]

        if use_batch_norm:
            modules.append(
                ("cnn1_bn", nn.BatchNorm2d(
                    cnn_out_channels[0],
                    affine=False,
                    track_running_stats=bn_track_running_stats)))

        modules += [
            ("cnn1_relu", nn.ReLU(inplace=True)),

            # -------------
            # Conv Block
            # -------------

            ("vdrop_cnn2", VDropConv2d(cnn_out_channels[0],
                                 cnn_out_channels[1],
                                 kernel_size, vdrop_data)),
            ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
        ]

        if use_batch_norm:
            modules.append(
                ("cnn2_bn", nn.BatchNorm2d(
                    cnn_out_channels[1],
                    affine=False,
                    track_running_stats=bn_track_running_stats)))

        modules += [
            ("cnn2_relu", nn.ReLU(inplace=True)),
            ("flatten", nn.Flatten()),

            # -------------
            # Linear Block
            # -------------

            ("vdrop_fc1", VDropLinear(
                (feature_map_sidelength**2) * cnn_out_channels[1],
                linear_units, vdrop_data)),
        ]
        if use_batch_norm:
            modules.append(
                ("fc1_bn", nn.BatchNorm1d(
                    linear_units,
                    affine=False,
                    track_running_stats=bn_track_running_stats)))

        modules += [
            ("fc1_relu", nn.ReLU(inplace=True)),

            # -------------
            # Output Layer
            # -------------

            ("vdrop_fc2", VDropLinear(linear_units, num_classes, vdrop_data)),
        ]
        super().__init__(OrderedDict(modules))
        vdrop_data.finalize()
        self.vdrop_data = vdrop_data

    def forward(self, *args, **kwargs):
        self.vdrop_data.compute_forward_data()
        ret = super().forward(*args, **kwargs)
        self.vdrop_data.clear_forward_data()
        return ret

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.vdrop_data = self.vdrop_data.to(*args, **kwargs)
        return ret


gsc_lenet_vdrop = partial(
    VDropLenet,
    input_size=(1, 32, 32),
    num_classes=12)