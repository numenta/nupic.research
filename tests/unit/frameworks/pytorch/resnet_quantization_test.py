#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
import copy
import unittest

import pytest
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni

from nupic.research.frameworks.pytorch.imagenet.network_utils import create_model
from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.models.sparse_resnets import resnet34, resnet50

TEST_MODEL_CLASS = [resnet34, resnet50]


def _create_test_model(model_class):
    model_args = dict(config=dict(num_classes=3, defaults_sparse=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_class=model_class,
        model_args=model_args,
        init_batch_norm=False,
        device=device,
    )
    return model


@pytest.mark.parametrize("model_class", TEST_MODEL_CLASS)
def test_fuse_model_conv_bn(model_class):
    original = _create_test_model(model_class=model_class)

    conv_layers = {
        name
        for name, module in original.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    bn_layers = {
        name
        for name, module in original.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    }

    # Fuse conv and bn only
    fused = copy.deepcopy(original)
    fused.fuse_model(fuse_relu=False)

    # Check if BN layers were removed
    assert all(
        isinstance(module, nn.Identity)
        for name, module in fused.named_modules()
        if name in bn_layers
    )

    # Check if all Conv/BN were merged
    conv_bn_layers = {
        name
        for name, module in fused.named_modules()
        if isinstance(module, nni.ConvBn2d)
    }
    assert conv_layers == conv_bn_layers

    # Validate output
    assert compare_models(original, fused, (3, 224, 224))


@pytest.mark.parametrize("model_class", TEST_MODEL_CLASS)
def test_fuse_model_conv_bn_relu(model_class):
    original = _create_test_model(model_class=model_class)

    conv_layers = {
        name
        for name, module in original.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    bn_layers = {
        name
        for name, module in original.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    }

    # Get all ReLU except for "post_activation"
    relu_layers = {
        name
        for name, module in original.named_modules()
        if isinstance(module, nn.ReLU) and "post_activation" not in name
    }

    # Fuse conv, bn and relu
    fused = copy.deepcopy(original)
    fused.fuse_model(fuse_relu=True)

    # Check if BN+ReLU layers were removed
    assert all(
        isinstance(module, nn.Identity)
        for name, module in fused.named_modules()
        if name in bn_layers | relu_layers
    )

    # Check if all Conv/BN/Relu were merged
    conv_bn_layers = {
        name
        for name, module in fused.named_modules()
        if isinstance(module, (nni.ConvBn2d, nni.ConvBnReLU2d))
    }
    assert conv_layers == conv_bn_layers

    # Validate output
    assert compare_models(original, fused, (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
