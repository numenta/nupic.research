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

import pickle

import torch
import torch.nn as nn

import nupic.torch


def fold_batchnorm_conv(conv2d, bn_2d):
    """
    Given a conv2d and its associated batchNorm2D, change the weights of
    the conv so that batch norm is no longer required for inference.
    """
    t = (bn_2d.running_var + bn_2d.eps).sqrt()
    conv2d.bias.data = (conv2d.bias - bn_2d.running_mean) / t
    t = t.reshape((conv2d.out_channels, 1, 1, 1))
    conv2d.weight.data = conv2d.weight / t


def fold_batchnorm_linear(linear, bn_linear):
    """
    Given a conv2d and its associated batchNorm2D, change the weights of
    the conv so that batch norm is no longer required for inference.
    """
    t = (bn_linear.running_var + bn_linear.eps).sqrt()
    linear.bias.data = (linear.bias - bn_linear.running_mean) / t
    t = t.reshape((linear.out_features, 1))
    linear.weight.data = linear.weight / t


def remove_batchnorm(model):
    """
    Return a new model that is equivalent to model, but with batch norm layers removed.

    Note: there are lots of restrictions to the structure of the model. We assume that
    batchnorm is applied right after conv or linear layers, before relu, maxpool,
    or kwinners.

    We don't currently support affine batchnorm (i.e. with gamma), but that should be a
    straightforward extension:
    https://stackoverflow.com/questions/49536856/
                tensorflow-how-to-merge-batchnorm-into-convolution-for-faster-inference

    https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
    https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/2

    Deleting a layer:
    https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/

    :param model:
    :return:
    """
    # Seems to be the best way to really ensure you're getting a copy
    modelr = pickle.loads(pickle.dumps(model))

    children = list(modelr.children())
    names = list(modelr._modules.keys())
    new_model = nn.Sequential()

    modelr.eval()
    with torch.no_grad():
        last_module_with_weights = None
        last_module_with_weights_type = None
        for i, module in enumerate(children):

            if ((type(module) == nn.modules.conv.Conv2d)
                    or (type(module)
                        == nupic.torch.modules.sparse_weights.SparseWeights2d)):
                last_module_with_weights = module
                last_module_with_weights_type = type(module)
                new_model.add_module(names[i], module)
            elif type(module) == nn.modules.batchnorm.BatchNorm2d:
                if last_module_with_weights_type == nn.modules.conv.Conv2d:
                    fold_batchnorm_conv(last_module_with_weights, module)
                elif (last_module_with_weights_type
                      == nupic.torch.modules.sparse_weights.SparseWeights2d):
                    fold_batchnorm_conv(last_module_with_weights.module, module)
                    last_module_with_weights.rezero_weights()

            elif ((type(module) == nn.modules.linear.Linear)
                  or (type(module)
                      == nupic.torch.modules.sparse_weights.SparseWeights)):
                last_module_with_weights = module
                last_module_with_weights_type = type(module)
                new_model.add_module(names[i], module)
            elif type(module) == nn.modules.batchnorm.BatchNorm1d:
                if last_module_with_weights_type == nn.modules.linear.Linear:
                    fold_batchnorm_linear(last_module_with_weights, module)
                elif (last_module_with_weights_type
                      == nupic.torch.modules.sparse_weights.SparseWeights):
                    fold_batchnorm_linear(last_module_with_weights.module, module)
                    last_module_with_weights.rezero_weights()
                else:
                    raise AssertionError

            # Everything else gets added back as is
            else:
                new_model.add_module(names[i], module)

    return new_model
