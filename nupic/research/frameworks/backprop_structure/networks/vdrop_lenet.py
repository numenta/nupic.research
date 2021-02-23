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
    VDropCentralData,
    MaskedVDropCentralData,
)

from nupic.research.frameworks.backprop_structure.modules.common_layers import (
    prunable_vdrop_linear,
    prunable_vdrop_conv2d,
)


class VDropLeNet(nn.Module):

    def __init__(self, input_shape=(1, 32, 32),
                 cnn_out_channels=(64, 64),
                 num_classes=12,
                 use_batch_norm=True,
                 z_logvar_init=-10,
                 vdrop_data_class=VDropCentralData,
                 kernel_size=5,
                 linear_units=1000,
                 maxpool_stride=2,
                 bn_track_running_stats=True,
                 conv_target_density = (1.0, 1.0),
                 linear_target_density  = (1.0, 1.0),
                 ):
        feature_map_sidelength = (
            (((input_shape[1] - kernel_size + 1) / maxpool_stride)
             - kernel_size + 1) / maxpool_stride
        )
        vdrop_data = vdrop_data_class(z_logvar_init=z_logvar_init)
        super().__init__()
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        modules = [
            # -------------
            # Conv Block
            # -------------

            ("vdrop_cnn1", prunable_vdrop_conv2d(input_shape[0],
                                 cnn_out_channels[0],
                                 kernel_size,
                                 vdrop_data,
                                 target_density=conv_target_density[0])),
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

            ("vdrop_cnn2", prunable_vdrop_conv2d(cnn_out_channels[0],
                                 cnn_out_channels[1],
                                 kernel_size,
                                 vdrop_data,
                                 target_density=conv_target_density[1])),
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

            ("vdrop_fc1", prunable_vdrop_linear(
                (feature_map_sidelength**2) * cnn_out_channels[1],
                linear_units,
                vdrop_data,
                target_density=linear_target_density[0])),
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

            ("vdrop_fc2", prunable_vdrop_linear(linear_units,
                                                num_classes,
                                                vdrop_data,
                                                target_density=linear_target_density[1])),
        ]
        self.classifier = nn.Sequential(OrderedDict(modules))
        vdrop_data.finalize()
        self.vdrop_data = vdrop_data

    def forward(self, *args, **kwargs):
        self.vdrop_data.compute_forward_data()
        ret = self.classifier.forward(*args, **kwargs)
        self.vdrop_data.clear_forward_data()
        return ret

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.vdrop_data = self.vdrop_data.to(*args, **kwargs)
        return ret


gsc_lenet_vdrop = partial(
    VDropLeNet,
    input_shape=(1, 32, 32),
    num_classes=12
)

gsc_lenet_vdrop_sparse = partial(
    VDropLeNet,
    vdrop_data_class = MaskedVDropCentralData,
    input_shape=(1, 32, 32),
    num_classes=12,
    linear_target_density=(0.1, 0.1),
    conv_target_density = (0.1, 0.1)
)

gsc_lenet_vdrop_super_sparse = partial(
    VDropLeNet,
    vdrop_data_class = MaskedVDropCentralData,
    input_shape=(1, 32, 32),
    num_classes=12,
    linear_target_density=(0.01, 0.01),
    conv_target_density = (0.01, 0.01)
)