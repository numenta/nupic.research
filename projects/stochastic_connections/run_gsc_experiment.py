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
from pathlib import Path

import numpy as np
import ray
import torch
import torch.utils.data
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

DATAPATH = Path(os.path.expanduser("~/nta/datasets/GSC"))

FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000


def get_train_filename(iteration):
    return "gsc_train{}.npz".format(iteration)


def preprocessed_dataset(filepath):
    """
    Get a processed dataset

    :param cachefilepath:
    Path to the processed data.
    :type cachefilepath: pathlib.Path

    :return: torch.utils.data.TensorDataset
    """
    x, y = np.load(filepath).values()
    x, y = map(torch.tensor, (x, y))

    return torch.utils.data.TensorDataset(x, y)


class StochasticGSCExperiment(tune.Trainable):
    def _setup(self, config):
        l0_strength = config["l0_strength"]
        l2_strength = config["l2_strength"]
        droprate_init = config["droprate_init"]
        self.use_tqdm = config["use_tqdm"]
        self.model_type = config["model_type"]

        self.val_loader = torch.utils.data.DataLoader(
            preprocessed_dataset(DATAPATH / "gsc_valid.npz"),
            batch_size=VALID_BATCH_SIZE,
            pin_memory=torch.cuda.is_available())

        num_classes = 12
        input_size = (1, 32, 32)

        cnn_out_channels = (64, 64)
        linear_units = 1000

        l0_strengths = (l0_strength, l0_strength, l0_strength, l0_strength)

        kernel_size = 5
        maxpool_stride = 2
        feature_map_sidelength = (
            (((input_size[1] - kernel_size + 1) / maxpool_stride)
             - kernel_size + 1) / maxpool_stride
        )
        assert(feature_map_sidelength == int(feature_map_sidelength))
        feature_map_sidelength = int(feature_map_sidelength)

        model_type = config["model_type"]
        learn_weight = config["learn_weight"]
        if model_type == "HardConcrete":
            temperature = 2 / 3
            self.model = nn.Sequential(OrderedDict([
                ("cnn1", HardConcreteGatedConv2d(
                    input_size[0], cnn_out_channels[0], kernel_size,
                    droprate_init=droprate_init, temperature=temperature,
                    l2_strength=l2_strength, l0_strength=l0_strengths[0],
                    learn_weight=learn_weight)),
                ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0], affine=False)),
                ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn1_relu", nn.ReLU()),
                ("cnn2", HardConcreteGatedConv2d(
                    cnn_out_channels[0], cnn_out_channels[1], kernel_size,
                    droprate_init=droprate_init, temperature=temperature,
                    l2_strength=l2_strength, l0_strength=l0_strengths[1],
                    learn_weight=learn_weight)),
                ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[1], affine=False)),
                ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn2_relu", nn.ReLU()),
                ("flatten", Flatten()),
                ("fc1", HardConcreteGatedLinear(
                    (feature_map_sidelength**2) * cnn_out_channels[1],
                    linear_units, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[2],
                    temperature=temperature, learn_weight=learn_weight)),
                ("fc1_bn", nn.BatchNorm1d(linear_units, affine=False)),
                ("fc1_relu", nn.ReLU()),
                ("fc2", HardConcreteGatedLinear(
                    linear_units, num_classes, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[3],
                    temperature=temperature, learn_weight=learn_weight)),
            ]))
        elif model_type == "Binary":
            self.model = nn.Sequential(OrderedDict([
                ("cnn1", BinaryGatedConv2d(
                    input_size[0], cnn_out_channels[0], kernel_size,
                    droprate_init=droprate_init, l2_strength=l2_strength,
                    l0_strength=l0_strengths[0], learn_weight=learn_weight)),
                ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0], affine=False)),
                ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn1_relu", nn.ReLU()),
                ("cnn2", BinaryGatedConv2d(
                    cnn_out_channels[0], cnn_out_channels[1], kernel_size,
                    droprate_init=droprate_init, l2_strength=l2_strength,
                    l0_strength=l0_strengths[1], learn_weight=learn_weight)),
                ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[0], affine=False)),
                ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn2_relu", nn.ReLU()),
                ("flatten", Flatten()),
                ("fc1", BinaryGatedLinear(
                    (feature_map_sidelength**2) * cnn_out_channels[1],
                    linear_units, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[2],
                    learn_weight=learn_weight)),
                ("fc1_bn", nn.BatchNorm1d(linear_units, affine=False)),
                ("fc1_relu", nn.ReLU()),
                ("fc2", BinaryGatedLinear(
                    linear_units, num_classes, droprate_init=droprate_init,
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
        batch_size = (FIRST_EPOCH_BATCH_SIZE if self.iteration == 0
                      else TRAIN_BATCH_SIZE)
        train_loader = torch.utils.data.DataLoader(
            preprocessed_dataset(
                DATAPATH / get_train_filename(self.iteration)),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        self.model.train()

        if self.use_tqdm:
            batches = tqdm(train_loader, leave=False, desc="Training")
        else:
            batches = train_loader

        train_loss = 0.
        train_correct = 0.

        for data, target in batches:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_function(output, target,
                                      batch_size / len(train_loader.dataset))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                train_loss += torch.sum(loss).item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()

            for layername in ["cnn1", "cnn2", "fc1", "fc2"]:
                layer = getattr(self.model, layername)
                layer.constrain_parameters()

        self.scheduler.step()

        self.model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            if self.use_tqdm:
                batches = tqdm(self.val_loader, leave=False, desc="Testing")
            else:
                batches = self.val_loader

            for data, target in batches:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += torch.sum(
                    self.loss_function(
                        output, target,
                        VALID_BATCH_SIZE / len(self.val_loader.dataset))).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        result = {
            "mean_accuracy": val_correct / len(self.val_loader.dataset),
            "mean_train_accuracy": train_correct / len(train_loader.dataset),
            "mean_loss": val_loss / len(self.val_loader.dataset),
            "mean_training_loss": train_loss / len(train_loader.dataset),
            "total_correct": val_correct,
            "lr": self.optimizer.state_dict()["param_groups"][0]["lr"],
        }
        result.update(self.nonzero_counts())

        expected_nz = 0
        inference_nz = 0
        flops = 0
        multiplies = 0
        for layername in ["cnn1", "cnn2", "fc1", "fc2"]:
            expected_nz += result[layername]["expected_nz"]
            inference_nz += result[layername]["inference_nz"]
            flops += result[layername]["flops"]
            multiplies += result[layername]["multiplies"]

        result["expected_nz"] = expected_nz
        result["inference_nz"] = inference_nz
        result["flops"] = flops
        result["multiplies"] = multiplies

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

            e_nz_by_unit = layer.get_expected_nonzeros()
            i_nz_by_unit = layer.get_inference_nonzeros()

            multiplies, adds = layer.count_inference_flops()

            result[layername] = {
                "hist_expected_nz_by_unit": e_nz_by_unit.tolist(),
                "expected_nz": torch.sum(e_nz_by_unit).item(),
                "hist_inference_nz_by_unit": layer.get_inference_nonzeros().tolist(),
                "inference_nz": torch.sum(i_nz_by_unit).item(),
                "num_input_units": num_inputs,
                "multiplies": multiplies,
                "flops": multiplies + adds,
            }

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HardConcrete",
                        choices=["HardConcrete", "Binary"])
    parser.add_argument("--lr", type=float, nargs="+", default=[0.01])
    # Suggested: 7e-4 for HardConcrete, 1e-4 for Binary
    parser.add_argument("--l0", type=float, nargs="+", default=[7e-4])
    parser.add_argument("--l2", type=float, nargs="+", default=[0])
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.9825])
    # Suggested: 0.5 for HardConcrete, 0.8 for Binary
    parser.add_argument("--droprate-init", type=float, nargs="+", default=[0.5])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--ray-address", type=str, default="localhost:6379")
    parser.add_argument("--fixedweight", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    # Verify that enough training data is present
    assert all((DATAPATH / get_train_filename(i)).exists()
               for i in range(args.epochs))

    if args.local:
        ray.init()
    else:
        ray.init(redis_address=args.ray_address)

    exp_name = "GSC-Stochastic"
    print("Running experiment {}".format(exp_name))
    analysis = tune.run(StochasticGSCExperiment,
                        name=exp_name,
                        num_samples=args.samples,
                        config={
                            "lr": tune.grid_search(args.lr),
                            # "l0_strength": tune.sample_from(
                            #     lambda _: math.exp(random.uniform(
                            #         math.log(5e-7), math.log(8e-5)))),
                            "l0_strength": tune.grid_search(args.l0),
                            "l2_strength": tune.grid_search(args.l2),
                            "model_type": tune.grid_search([args.model]),
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
