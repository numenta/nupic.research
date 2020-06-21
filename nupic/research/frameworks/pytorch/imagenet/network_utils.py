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
import io
import pickle
from collections import OrderedDict

import torch
import torchvision.models
import torchvision.models.resnet
from torch import nn as nn

import nupic.research
import nupic.research.frameworks.pytorch.models.resnets
from nupic.research.frameworks.pytorch.model_utils import deserialize_state_dict
from nupic.torch.compatibility import upgrade_to_masked_sparseweights


def init_resnet50_batch_norm(model):
    """
    Initialize ResNet50 batch norm modules
    See https://arxiv.org/pdf/1706.02677.pdf

    :param model: Resnet 50 model
    """
    for m in model.modules():
        if isinstance(m, torchvision.models.resnet.BasicBlock):
            # initialized the last BatchNorm in each BasicBlock to 0
            m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
        elif isinstance(m, torchvision.models.resnet.Bottleneck):
            # initialized the last BatchNorm in each Bottleneck to 0
            m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
        elif isinstance(m, (
            nupic.research.frameworks.pytorch.models.resnets.BasicBlock,
            nupic.research.frameworks.pytorch.models.resnets.Bottleneck
        )):
            # initialized the last BatchNorm in each BasicBlock to 0
            *_, last_bn = filter(lambda x: isinstance(x, nn.BatchNorm2d),
                                 m.regular_path)
            last_bn.weight = nn.Parameter(torch.zeros_like(last_bn.weight))
        elif isinstance(m, nn.Linear):
            # initialized linear layers weights from a gaussian distribution
            m.weight.data.normal_(0, 0.01)


def create_model(model_class, model_args, init_batch_norm, device,
                 checkpoint_file=None, detect_old_sparseweights=True):
    """
    Create imagenet experiment model with option to load state from checkpoint

    :param model_class:
            The model class. Must inherit from torch.nn.Module
    :param model_args:
        The model constructor arguments
    :param init_batch_norm:
        Whether or not to initialize batch norm modules
    :param device:
        Model device
    :param checkpoint_file:
        Optional checkpoint file to load model state

    :return: Configured model
    """
    model = model_class(**model_args)
    if init_batch_norm:
        init_resnet50_batch_norm(model)
    model.to(device)

    # Load model parameters from checkpoint
    if checkpoint_file is not None:
        with open(checkpoint_file, "rb") as pickle_file:
            state = pickle.load(pickle_file)
        with io.BytesIO(state["model"]) as buffer:
            state_dict = deserialize_state_dict(buffer, device)

        if detect_old_sparseweights:
            state_dict = upgrade_to_masked_sparseweights(state_dict)

        # Make sure checkpoint is compatible with model
        if model.state_dict().keys() != state_dict.keys():
            state_dict = OrderedDict(
                zip(model.state_dict().keys(), state_dict.values()))

        model.load_state_dict(state_dict)

    return model
