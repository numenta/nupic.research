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

from .le_sparse_net import LeSparseNet
from .mobilenetv1 import (
    MobileNetV1,
    mobile_net_v1_sparse_depth,
    mobile_net_v1_sparse_point,
    separable_convolution2d,
)
from .not_so_densenet import DenseNetCIFAR, NoSoDenseNetCIFAR
from .resnet_models import ResNet, resnet9
from .vgg_sparse_net import VGGSparseNet, vgg19_dense_net, vgg19_sparse_net
from .pretrained_models import (
    resnext101_32x48d_wsl,
    resnext101_32x16d_wsl,
    resnet50_swsl,
    resnext50_32x4d_swsl,
    se_resnet50,
    se_resnext50_32x4d,
    xception
)
from .common import StandardMLP