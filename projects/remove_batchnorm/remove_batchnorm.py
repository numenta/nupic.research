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
import random

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from projects.remove_batchnorm.simple_net import SimpleCNN

def train(model, num_samples=20):
    """
    Train the model on random inputs.

    :param model: pytorch model to be trained

    :param loader: data to train on
    :param optimizer: Optimizer object used to train the model.

    :param criterion: loss function to use

    :param device:

    :type device: :class:`torch.device`
    """
    # Create a random training set
    x = torch.randn((num_samples,) + (1, 32, 32))
    targets = torch.randint(0, 12, (num_samples,))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.2)
    model.train()
    for batch_idx in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()
        print(loss.item())

def inspect_model(model):
    model.eval()
    with torch.no_grad():
        print(model)
        bnc = model._modules["cnn1_batchnorm"]
        print(bnc)
        bnl = model._modules["linear_bn"]
        print(bnl)
        print("running_mean", bnl.running_mean)
        print("running_var", bnl.running_var)
        # print()

        # From https://discuss.pytorch.org/t/affine-parameter-in-batchnorm/6005/2
        z = torch.zeros((1, 3))
        print("zeros", bnl(z))
        print("zeros transformed", (z-bnl.running_mean)/(bnl.running_var + bnl.eps).sqrt())

        o = torch.ones((1, 3))
        print("ones", bnl(o))
        print("ones transformed", (o-bnl.running_mean)/(bnl.running_var + bnl.eps).sqrt())
        print("---------------------\n\n")

        # With CNN
        cnn1 = model._modules["cnn1"]
        z = torch.zeros((1, 1, 6, 6))
        o = torch.ones((1, 1, 6, 6))
        cz = cnn1(z)
        co = cnn1(o)
        print("cnn(zeros)", cz)
        print("bn(cnn(zeros))", bnc(cnn1(z)))
        print("cnn(ones)", co)
        print("bn(cnn(ones))", bnc(cnn1(o)))

        print("cnn1.bias", cnn1.bias)
        t = (bnc.running_var + bnc.eps).sqrt()
        cnn1.bias.data = (cnn1.bias - bnc.running_mean) / t
        t = t.reshape((2,1,1,1))
        cnn1.weight.data = cnn1.weight / t
        cz = cnn1(z)
        co = cnn1(o)
        print("cnn(zeros) after rescaling", cz)
        print("cnn(ones) after rescaling", co)

    pass

def remove_batchnorm(model):
    """
    https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
    https://discuss.pytorch.org/t/how-to-replace-all-relu-activations-in-a-pretrained-network/31591/2

    :param model:
    :return:
    """
    modelr = pickle.loads(pickle.dumps(model))

    bnc = modelr._modules["cnn1_batchnorm"]
    pass

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SimpleCNN()
    train(net)
    inspect_model(net)
    remove_batchnorm(net)
