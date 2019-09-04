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
Run experiments with L0 regularization and check the resulting sparsity and
benchmark performance
"""

import argparse
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import ray
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

from nupic.torch.modules import Flatten
from reparameterization_layers import (
    StochasticConnectionsConv2d,
    StochasticConnectionsLinear,
)


class StochasticExperiment(object):
    def __init__(self, input_size, num_classes, l0_strength, l2_strength):
        conv_dims = (20, 50)
        fc_dims = 500

        temperature = 2 / 3
        l0_strengths = (l0_strength, l0_strength, l0_strength, l0_strength)

        kernel_sidelength = 5
        maxpool_stride = 2
        feature_map_sidelength = (
            (((input_size[1] - kernel_sidelength + 1) / maxpool_stride)
             - kernel_sidelength + 1) / maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        self.model = nn.Sequential(OrderedDict([
            ("cnn1", StochasticConnectionsConv2d(
                input_size[0], conv_dims[0], kernel_sidelength,
                droprate_init=0.5, temperature=temperature,
                l2_strength=l2_strength, l0_strength=l0_strengths[0])),
            ("cnn1_relu", nn.ReLU()),
            ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("cnn2", StochasticConnectionsConv2d(
                conv_dims[0], conv_dims[1], kernel_sidelength,
                droprate_init=0.5, temperature=temperature,
                l2_strength=l2_strength, l0_strength=l0_strengths[1])),
            ("cnn2_relu", nn.ReLU()),
            ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
            ("flatten", Flatten()),
            ("fc1", StochasticConnectionsLinear(
                (feature_map_sidelength**2) * conv_dims[1], fc_dims,
                droprate_init=0.5, l2_strength=l2_strength,
                l0_strength=l0_strengths[2], temperature=temperature)),
            ("fc1_relu", nn.ReLU()),
            ("fc2", StochasticConnectionsLinear(
                fc_dims, num_classes, droprate_init=0.5,
                l2_strength=l2_strength, l0_strength=l0_strengths[3],
                temperature=temperature)),
        ]))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.device = device

        self.loglike = nn.CrossEntropyLoss().to(self.device)

        lr = 0.001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def train(self, loader):
        self.model.train()

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_function(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for layername in ["cnn1", "cnn2", "fc1", "fc2"]:
                layer = getattr(self.model, layername)
                layer.constrain_parameters()

    def test(self, loader):
        self.model.eval()
        loss = 0
        total_correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # sum up batch loss
                loss += torch.sum(self.loss_function(output, target)).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        return {"accuracy": total_correct / len(loader.dataset),
                "loss": loss / len(loader.dataset),
                "total_correct": total_correct}

    def loss_function(self, output, target):
        return self.loglike(output, target) + self.regularization()

    def regularization(self):
        reg = torch.tensor(0.).to(self.device)
        for layername in ["cnn1", "cnn2", "fc1", "fc2"]:
            layer = getattr(self.model, layername)
            reg += - layer.regularization()
        return reg

    def nonzero_counts(self):
        result = {}
        for layername in ["cnn1", "cnn2", "fc1", "fc2"]:
            layer = getattr(self.model, layername)

            # Measure two different types of nonzeros:
            e_gates = 1 - layer.cdf_qz(0)
            e_nz_by_unit = e_gates.sum(
                dim=tuple(range(1, len(e_gates.shape))))
            inf_gates = F.hardtanh(
                torch.sigmoid(layer.loga) * (1.1 - -.1) + -.1,
                min_val=0, max_val=1)
            inf_nz_by_unit = (inf_gates > 0).sum(
                dim=tuple(range(1, len(inf_gates.shape))))

            num_inputs = 1.
            for d in layer.weight.size()[1:]:
                num_inputs *= d

            result[layername] = {
                "expected_nz_by_unit": e_nz_by_unit.detach().numpy(),
                "inference_nz_by_unit": inf_nz_by_unit.detach().numpy(),
                "num_input_units": num_inputs,
            }

        return result


@ray.remote
class MNISTExperiment(object):
    def __init__(self, l0_strength, l2_strength):
        batch_size = 100
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST("./data", train=True, download=True,
                           transform=transform),
            batch_size=batch_size, shuffle=True, num_workers=4,
            pin_memory=torch.cuda.is_available())
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST("./data", train=False, transform=transform),
            batch_size=batch_size, num_workers=4,
            pin_memory=torch.cuda.is_available())
        num_classes = 10
        input_size = (1, 28, 28)

        self.exp = StochasticExperiment(input_size, num_classes, l0_strength,
                                        l2_strength)

    def step(self, train=True):
        if train:
            self.exp.train(self.train_loader)
        results = self.exp.test(self.val_loader)
        nonzeros = self.exp.nonzero_counts()
        return (results, nonzeros)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("foldername", type=str)
    parser.add_argument("--l0", type=float, default=0.00002)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--redis-address", type=str, default=None)

    args = parser.parse_args()

    cwd = Path(os.path.dirname(os.path.realpath(__file__)))
    folderpath = cwd / args.foldername

    os.makedirs(folderpath, exist_ok=True)

    ray.init(redis_address=args.redis_address)

    print("Saving results to {}".format(folderpath))
    outpath = folderpath / "config.pkl"
    with open(outpath, "wb") as f:
        pickle.dump({"l0_strength": args.l0,
                     "l2_strength": args.l2}, f)

    num_digits = len(str(int(args.epochs)))
    outfilefmt = "res{:0" + str(num_digits) + "d}.pkl"

    exp = MNISTExperiment.remote(args.l0, args.l2)
    results, nonzeros = ray.get(exp.step.remote(train=False))
    print("Initial epoch: {}".format(results))
    outpath = folderpath / outfilefmt.format(0)
    with open(outpath, "wb") as f:
        pickle.dump((results, nonzeros), f)

    for epoch in tqdm(range(1, args.epochs + 1), leave=False,
                      desc="Running remotely", unit="epoch"):
        results, nonzeros = ray.get(exp.step.remote())
        print("Epoch {}: {}".format(epoch, results))
        outpath = folderpath / outfilefmt.format(epoch)
        with open(outpath, "wb") as f:
            pickle.dump((results, nonzeros), f)
