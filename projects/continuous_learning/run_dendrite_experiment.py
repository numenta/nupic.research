# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cont_speech_experiment import ContinuousSpeechExperiment
from nupic.research.frameworks.continuous_learning.dendrite_layers import DendriteLayer
from nupic.research.frameworks.continuous_learning.utils import freeze_output_layer
from nupic.research.frameworks.pytorch.model_utils import evaluate_model
from nupic.research.support import parse_config
from nupic.torch.modules import Flatten, KWinners2d, SparseWeights, SparseWeights2d

config_file = "experiments.cfg"
with open(config_file) as cf:
    config_init = parse_config(cf)

exp = "sparseCNN2"

#  play with the config values here
config = config_init[exp]
config["name"] = exp
config["use_dendrites"] = True
config["use_batch_norm"] = False
config["cnn_out_channels"] = (64, 64)
config["cnn_percent_on"] = (0.12, 0.07)
config["cnn_weight_sparsity"] = (0.15, 0.05)
config["dendrites_per_cell"] = 2
config["batch_size"] = 64


def get_experiment():
    """ The experiment class here is
    just to easily access the DataLoaders """
    experiment = ContinuousSpeechExperiment(config=config)
    return experiment


def get_no_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def clear_labels(labels, n_classes=5):
    indices = np.arange(n_classes)
    out = np.delete(indices, labels)
    return out


class ToyNetwork(nn.Module):
    """ Toy network; here dpc is dendrites_per_neuron """
    def __init__(
        self,
        dpc=3,
        cnn_w_sparsity=0.05,
        linear_w_sparsity=0.5,
        cat_w_sparsity=0.01,
        n_classes=4,
    ):
        super(ToyNetwork, self).__init__()
        conv_channels = 128
        self.n_classes = n_classes
        self.conv1 = SparseWeights2d(
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=10,
                padding=0,
                stride=1,
            ),
            cnn_w_sparsity,
        )
        self.kwin1 = KWinners2d(conv_channels, percent_on=0.1)
        self.bn = nn.BatchNorm2d(conv_channels, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.flatten = Flatten()

        self.d1 = DendriteLayer(
            in_dim=int(conv_channels / 64) * 7744,
            out_dim=1000,
            dendrites_per_neuron=dpc,
        )

        self.linear = SparseWeights(nn.Linear(1000, n_classes + 1), linear_w_sparsity)

        self.cat = SparseWeights(nn.Linear(n_classes + 1, 1000 * dpc), cat_w_sparsity)

    def forward(self, x, label=None, batch_norm=False):
        y = self.conv1(x)
        if batch_norm:
            y = self.bn(y)
        y = self.kwin1(self.mp1(y))
        y = self.flatten(y)
        if label is not None:
            yhat = torch.eye(self.n_classes + 1)[label]
            y = self.d1(y, torch.tanh(self.cat(yhat.cuda())))
        else:
            y = self.d1(y)
        y = F.log_softmax(self.linear(y), dim=1)
        return y


def train_full(categorical=False):
    experiment = get_experiment()
    net = ToyNetwork(dpc=1, cnn_w_sparsity=0.1).cuda()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)  # weight_decay=0.)
    criterion = F.nll_loss

    loader = experiment.full_train_loader
    for x, y in loader:
        opt.zero_grad()
        if categorical:
            out = net(x.cuda(), y.cuda())
        else:
            out = net(x.cuda())  # no categorical projection
        loss = criterion(out, y.cuda())
        loss.backward()
        opt.step()

    acc_ = evaluate_model(net, experiment.gen_test_loader, torch.device("cuda"))[
        "mean_accuracy"
    ]
    print("Accuracy: {}".format(np.round(acc_, 2)))
    return acc_


def train_sequential(
    categorical=False,
    dpc=1,
    cnn_weight_sparsity=0.1,
    linear_w_sparsity=0.5,
    cat_w_sparsity=0.01,
    optim="Adam",
):
    experiment = get_experiment()

    net = ToyNetwork(
        dpc=dpc,
        cnn_w_sparsity=cnn_weight_sparsity,
        linear_w_sparsity=linear_w_sparsity,
        cat_w_sparsity=cat_w_sparsity,
    ).cuda()

    if optim == "Adam":
        opt = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0)

    criterion = F.nll_loss

    train_inds = np.arange(1, 5).reshape(2, 2)
    losses = []
    for i in range(len(train_inds)):
        experiment.combine_classes(train_inds[i])
        loader = experiment.train_loader
        for x, y in loader:
            opt.zero_grad()
            if categorical:
                out = net(x.cuda(), y.cuda())
            else:
                out = net(x.cuda())  # no categorical projection

            loss = criterion(out, y.cuda())
            loss.backward()
            losses.append(loss.detach().cpu().numpy())

            freeze_output_layer(
                net, clear_labels(train_inds), layer_type="kwinner", linear_number=""
            )

            opt.step()
        acc_ = [
            np.round(
                evaluate_model(net, experiment.test_loader[k], torch.device("cuda"))[
                    "mean_accuracy"
                ],
                2,
            )
            for k in train_inds[i]
        ]
        print(acc_)

    full_acc = [
        np.round(
            evaluate_model(net, experiment.test_loader[k], torch.device("cuda"))[
                "mean_accuracy"
            ],
            2,
        )
        for k in train_inds.flatten()
    ]

    print("Categorical: {}, acc={}".format(categorical, full_acc))
    return full_acc


if __name__ == "__main__":
    train_sequential(categorical=False, dpc=2, cat_w_sparsity=0.1)
    train_sequential(categorical=True, dpc=2, cat_w_sparsity=0.1)
