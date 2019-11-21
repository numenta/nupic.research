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
from collections import OrderedDict

import ray
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ray import tune
from ray.tune.logger import CSVLogger, JsonLogger
from torch import nn
from tqdm import tqdm

from nupic.research.frameworks.pytorch.tf_tune_utils import TFLoggerPlus
from nupic.research.frameworks.stochastic_connections.binary_layers import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
)
from nupic.research.frameworks.stochastic_connections.reparameterization_layers import (
    HardConcreteGatedConv2d,
    HardConcreteGatedLinear,
)
from nupic.torch.modules import Flatten


class StochasticMNISTExperiment(tune.Trainable):
    def _setup(self, config):
        l0_strength = config["l0_strength"]
        l2_strength = config["l2_strength"]
        droprate_init = config["droprate_init"]
        self.use_tqdm = config["use_tqdm"]

        data_path = os.path.expanduser("~/nta/datasets")
        self.batch_size = 100
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=True, download=True,
                           transform=transform),
            batch_size=self.batch_size, shuffle=True, num_workers=4,
            pin_memory=torch.cuda.is_available())
        self.val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train=False, transform=transform),
            batch_size=self.batch_size, num_workers=4,
            pin_memory=torch.cuda.is_available())
        num_classes = 10
        input_size = (1, 28, 28)

        conv_dims = (20, 50)
        fc_dims = 500

        l0_strengths = (l0_strength, l0_strength, l0_strength, l0_strength)

        kernel_sidelength = 5
        maxpool_stride = 2
        feature_map_sidelength = (
            (((input_size[1] - kernel_sidelength + 1) / maxpool_stride)
             - kernel_sidelength + 1) / maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        model_type = config["model_type"]
        learn_weight = config["learn_weight"]
        if model_type == "HardConcrete":
            temperature = 2 / 3
            self.model = nn.Sequential(OrderedDict([
                ("cnn1", HardConcreteGatedConv2d(
                    input_size[0], conv_dims[0], kernel_sidelength,
                    droprate_init=droprate_init, temperature=temperature,
                    l2_strength=l2_strength, l0_strength=l0_strengths[0],
                    learn_weight=learn_weight)),
                ("cnn1_relu", nn.ReLU()),
                ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn2", HardConcreteGatedConv2d(
                    conv_dims[0], conv_dims[1], kernel_sidelength,
                    droprate_init=droprate_init, temperature=temperature,
                    l2_strength=l2_strength, l0_strength=l0_strengths[1],
                    learn_weight=learn_weight)),
                ("cnn2_relu", nn.ReLU()),
                ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("flatten", Flatten()),
                ("fc1", HardConcreteGatedLinear(
                    (feature_map_sidelength**2) * conv_dims[1], fc_dims,
                    droprate_init=droprate_init, l2_strength=l2_strength,
                    l0_strength=l0_strengths[2], temperature=temperature,
                    learn_weight=learn_weight)),
                ("fc1_relu", nn.ReLU()),
                ("fc2", HardConcreteGatedLinear(
                    fc_dims, num_classes, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[3],
                    temperature=temperature, learn_weight=learn_weight)),
            ]))
        elif model_type == "Binary":
            self.model = nn.Sequential(OrderedDict([
                ("cnn1", BinaryGatedConv2d(
                    input_size[0], conv_dims[0], kernel_sidelength,
                    droprate_init=droprate_init, l2_strength=l2_strength,
                    l0_strength=l0_strengths[0], learn_weight=learn_weight)),
                ("cnn1_relu", nn.ReLU()),
                ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn2", BinaryGatedConv2d(
                    conv_dims[0], conv_dims[1], kernel_sidelength,
                    droprate_init=droprate_init, l2_strength=l2_strength,
                    l0_strength=l0_strengths[1], learn_weight=learn_weight)),
                ("cnn2_relu", nn.ReLU()),
                ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("flatten", Flatten()),
                ("fc1", BinaryGatedLinear(
                    (feature_map_sidelength**2) * conv_dims[1], fc_dims,
                    droprate_init=droprate_init, l2_strength=l2_strength,
                    l0_strength=l0_strengths[2], learn_weight=learn_weight)),
                ("fc1_relu", nn.ReLU()),
                ("fc2", BinaryGatedLinear(
                    fc_dims, num_classes, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[3],
                    learn_weight=learn_weight)),
            ]))
        else:
            raise ValueError("Unrecognized model type: {}".format(model_type))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.device = device

        self.loglike = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), config["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                         gamma=config["gamma"])

    def _train(self):
        self.model.train()

        if self.use_tqdm:
            batches = tqdm(self.train_loader, leave=False, desc="Training")
        else:
            batches = self.train_loader

        for data, target in batches:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_function(
                output, target,
                self.batch_size / len(self.train_loader.dataset))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for layername in ["cnn1", "cnn2", "fc1", "fc2"]:
                layer = getattr(self.model, layername)
                layer.constrain_parameters()

        self.scheduler.step()

        self.model.eval()
        loss = 0
        total_correct = 0
        with torch.no_grad():
            if self.use_tqdm:
                batches = tqdm(self.val_loader, leave=False, desc="Testing")
            else:
                batches = self.train_loader

            for data, target in batches:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += torch.sum(
                    self.loss_function(
                        output, target,
                        self.batch_size / len(self.val_loader.dataset))
                ).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        result = {
            "mean_accuracy": total_correct / len(self.val_loader.dataset),
            "mean_loss": loss / len(self.val_loader.dataset),
            "total_correct": total_correct,
        }
        result.update(self.nonzero_counts())
        return result

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

    def loss_function(self, output, target, dataset_percent):
        loss = self.loglike(output, target)
        loss += dataset_percent * self.regularization()
        return loss

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

            num_inputs = 1.
            for d in layer.weight_size()[1:]:
                num_inputs *= d

            result[layername] = {
                "hist_expected_nz_by_unit": layer.get_expected_nonzeros().tolist(),
                "hist_inference_nz_by_unit": layer.get_inference_nonzeros().tolist(),
                "num_input_units": num_inputs,
            }

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HardConcrete",
                        choices=["HardConcrete", "Binary"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l0", type=float, nargs="+", default=[7e-4])
    parser.add_argument("--l2", type=float, nargs="+", default=[0])
    parser.add_argument("--gamma", type=float, nargs="+", default=[1.0])
    parser.add_argument("--droprate-init", type=float, nargs="+", default=[0.5])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--ray-address", type=str, default="localhost:6379")
    parser.add_argument("--fixedweight", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    if args.local:
        ray.init()
    else:
        ray.init(redis_address=args.ray_address)

    exp_name = "MNIST-Stochastic"
    print("Running experiment {}".format(exp_name))
    analysis = tune.run(StochasticMNISTExperiment,
                        name=exp_name,
                        num_samples=args.samples,
                        config={
                            "lr": args.lr,
                            "l0_strength": tune.grid_search(args.l0),
                            "l2_strength": tune.grid_search(args.l2),
                            "model_type": args.model,
                            "learn_weight": not args.fixedweight,
                            "use_tqdm": args.progress,
                            "gamma": tune.grid_search(args.gamma),
                            "droprate_init": tune.grid_search(args.droprate_init),
                        },
                        stop={"training_iteration": args.epochs},
                        checkpoint_at_end=True,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": (1 if torch.cuda.is_available() else 0)
                        },
                        loggers=(JsonLogger, CSVLogger, TFLoggerPlus),
                        verbose=1)

    print(("To browse results, instantiate "
           '`tune.Analysis("~/ray_results/{}")`').format(exp_name))
