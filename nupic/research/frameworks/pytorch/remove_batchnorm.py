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
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.quantization.fuse_modules import fuse_modules, fuse_known_modules


def fold_batchnorm_conv(conv2d, bn_2d):
    """
    Given a conv2d layer and its associated BatchNorm2d, returns a copy of the
    original layer with the weights changed so that batch norm is no longer
    required for inference.
    """
    assert (not (conv2d.training or bn_2d.training)), \
        "This function should only be called during inference"

    if bn_2d.affine:
        bn_w = bn_2d.weight
        bn_b = bn_2d.bias
    else:
        bn_w = torch.ones(bn_2d.num_features)
        bn_b = torch.zeros(bn_2d.num_features)

    folded = copy.deepcopy(conv2d)
    t = (bn_2d.running_var + bn_2d.eps).rsqrt()
    folded.weight = nn.Parameter(conv2d.weight * (bn_w * t).reshape((-1, 1, 1, 1)))
    if conv2d.bias is not None:
        folded.bias = nn.Parameter((conv2d.bias - bn_2d.running_mean) * t * bn_w + bn_b)

    return folded


def fold_batchnorm_linear(linear, bn_linear):
    """
    Given a linear layer and its associated BatchNorm1d, returns a copy of the
    original layer with the weights changed so that batch norm is no longer
    required for inference.
    """
    assert (not (linear.training or bn_linear.training)), \
        "This function should only be called during inference"

    folded = copy.deepcopy(linear)
    t = (bn_linear.running_var + bn_linear.eps).rsqrt()
    folded.bias = nn.Parameter((linear.bias - bn_linear.running_mean) * t)
    folded.weight = nn.Parameter(linear.weight * t.reshape((-1, 1)))

    return folded


FUSE_MODULES_FUNCTIONS = {
    (nn.Conv2d, nn.BatchNorm2d): fold_batchnorm_conv,
    (nn.Linear, nn.BatchNorm1d): fold_batchnorm_linear,
}
FUSE_MODULES_TYPES = sum(FUSE_MODULES_FUNCTIONS.keys(), ())


def fuse_conv_linear_bn(mod_list):
    """
    Modified `torch.quantization.fuse_known_modules` adding support for linear,
    batch_norm modules.

    Handles the following sequence of modules::

        Conv2d, BatchNorm2d
        Linear, BatchNorm1d

    Everything else falls back to torch.quantization.fuse_known_modules

    .. seealso:: :func:`torch.quantization.fuse_known_modules` for more details
    """
    module_types = tuple(type(m) for m in mod_list)
    fuser_function = FUSE_MODULES_FUNCTIONS.get(module_types, None)
    if fuser_function is None:
        # Falls back to torch.quantization.fuse_known_modules
        return fuse_known_modules(mod_list)

    new_mod = [None] * len(mod_list)
    new_mod[0] = fuser_function(*mod_list)

    for i in range(1, len(mod_list)):
        new_mod[i] = nn.Identity()
        new_mod[i].training = mod_list[0].training

    return new_mod


def remove_batchnorm(model):
    """
    Return a new model that is equivalent to model, but with batch norm layers removed.

    Note: there are lots of restrictions to the structure of the model. We assume that
    batchnorm is applied right after conv or linear layers, before relu, maxpool,
    or kwinners.

    Deleting a layer:
    https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/

    :param model:
    :return:
    """

    # Create a list containing the names of all foldable layers in order of appearance
    modules_to_fuse = [x[0] for x in model.named_modules()
                       if isinstance(x[1], FUSE_MODULES_TYPES)]

    # Assuming every foldable layer is followed by a BatchNorm, group the modules
    # into a list of layer_name, bn_name tuples
    modules_to_fuse = list(zip(modules_to_fuse[0::2], modules_to_fuse[1::2]))

    model.eval()
    folded_model = fuse_modules(model, modules_to_fuse,
                                fuser_func=fuse_conv_linear_bn)
    # Remove Identity layers
    modules = OrderedDict([(name, module)
                           for name, module in folded_model.named_children()
                           if not isinstance(module, nn.Identity)])

    return nn.Sequential(modules)
