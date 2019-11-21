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
from functools import partial
from pathlib import Path

import numpy as np
import ray
import torch
import torch.utils.data
from ray import tune
from ray.tune.logger import CSVLogger, JsonLogger
from torch import nn
from tqdm import tqdm

import nupic.research.frameworks.stochastic_connections.binary_layers as bl
from nupic.research.frameworks.pytorch.tf_tune_utils import TFLoggerPlus
from nupic.research.frameworks.stochastic_connections.reparameterization_layers import (
    HardConcreteGatedConv2d,
    HardConcreteGatedLinear,
)
from nupic.torch.modules import Flatten, KWinners, KWinners2d, update_boost_strength

# import cProfile
# import pstats

DATAPATH = Path(os.path.expanduser("~/nta/datasets/GSC"))

VALID_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 1000


def get_train_filename(iteration):
    return "gsc_train{}.npz".format(iteration % 100)


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
        self.tqdm_mininterval = config["tqdm_mininterval"]
        self.model_type = config["model_type"]
        self.train_batch_size = config["batch_size"]
        self.train_first_batch_size = config["first_batch_size"]
        self.lr_decay_period = config["lr_decay_period"]

        self.val_loader = torch.utils.data.DataLoader(
            preprocessed_dataset(DATAPATH / "gsc_valid.npz"),
            batch_size=VALID_BATCH_SIZE,
            pin_memory=torch.cuda.is_available())

        Dropout = (partial(nn.Dropout, 0.5)  # noqa: N806
                   if config["use_dropout"]
                   else nn.Identity)
        ConvDropout = nn.Identity  # noqa: N806

        Nonlinearity = (  # noqa: N806
            nn.ReLU
            if config["activation"] == "ReLU" else
            partial(
                bl.HeavisideStep,
                neg1_activation=(config["activation"] == "BinaryNeg1"),
                cancel_gradient=(config["cancel_gradient"]),
                inplace=False))

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
        random_weight = config["random_weight"]
        affine = False
        bias = True
        deterministic = config["deterministic"]
        baseline_bias = (config["activation"] == "Binary0Corrected")
        self.optimize_inference = config["optimize_inference"]
        one_sample_per_item = config["one_sample_per_item"]

        if model_type == "HardConcrete":
            temperature = 2 / 3
            self.model = nn.Sequential(OrderedDict([
                ("cnn1", HardConcreteGatedConv2d(
                    input_size[0], cnn_out_channels[0], kernel_size,
                    droprate_init=droprate_init, temperature=temperature,
                    l2_strength=l2_strength, l0_strength=l0_strengths[0],
                    learn_weight=learn_weight, bias=bias)),
                ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0], affine=affine)),
                ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn1_relu", Nonlinearity()),
                ("cnn2", HardConcreteGatedConv2d(
                    cnn_out_channels[0], cnn_out_channels[1], kernel_size,
                    droprate_init=droprate_init, temperature=temperature,
                    l2_strength=l2_strength, l0_strength=l0_strengths[1],
                    learn_weight=learn_weight, bias=bias)),
                ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[1], affine=affine)),
                ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn2_relu", Nonlinearity()),
                ("flatten", Flatten()),
                ("fc1", HardConcreteGatedLinear(
                    (feature_map_sidelength**2) * cnn_out_channels[1],
                    linear_units, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[2],
                    temperature=temperature, learn_weight=learn_weight,
                    bias=bias)),
                ("fc1_bn", nn.BatchNorm1d(linear_units, affine=affine)),
                ("fc1_relu", Nonlinearity()),
                ("fc2", HardConcreteGatedLinear(
                    linear_units, num_classes, droprate_init=droprate_init,
                    l2_strength=l2_strength, l0_strength=l0_strengths[3],
                    temperature=temperature, learn_weight=learn_weight,
                    bias=bias)),
            ]))
            self.sparse_layers = ["cnn1", "cnn2", "fc1", "fc2"]
        elif model_type == "Alex":
            ratio_infl = config["ratio_infl"]

            self.model = nn.Sequential(OrderedDict([
                ("cnn1", nn.Conv2d(
                    input_size[0], 64 * ratio_infl, kernel_size, stride=2,
                    padding=1, bias=bias)),
                ("cnn1_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
                ("cnn1_bn", nn.BatchNorm2d(64 * ratio_infl, affine=affine)),
                ("cnn1_nonlin", nn.ReLU()),

                ("cnn2", nn.Conv2d(
                    64 * ratio_infl, 192 * ratio_infl, kernel_size=5, padding=2,
                    bias=bias)),
                ("cnn2_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
                ("cnn2_bn", nn.BatchNorm2d(192 * ratio_infl, affine=affine)),
                ("cnn2_nonlin", nn.ReLU()),

                ("cnn3", nn.Conv2d(
                    192 * ratio_infl, 384 * ratio_infl, kernel_size=3, padding=1)),
                ("cnn3_bn", nn.BatchNorm2d(384 * ratio_infl, affine=affine)),
                ("cnn3_nonlin", nn.ReLU()),

                ("cnn4", nn.Conv2d(
                    384 * ratio_infl, 256 * ratio_infl, kernel_size=3,
                    padding=1, bias=bias)),
                ("cnn4_bn", nn.BatchNorm2d(256 * ratio_infl, affine=affine)),
                ("cnn4_nonlin", nn.ReLU()),

                ("cnn5", nn.Conv2d(
                    256 * ratio_infl, 256, kernel_size=3, padding=1, bias=bias)),
                ("cnn5_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
                ("cnn5_bn", nn.BatchNorm2d(256, affine=affine)),
                ("cnn5_nonlin", nn.ReLU()),

                ("flatten", Flatten()),

                ("fc1", nn.Linear(256, 4096, bias=bias)),
                ("fc1_bn", nn.BatchNorm1d(4096, affine=affine)),
                ("fc1_nonlin", nn.ReLU()),
                ("fc1_dropout", nn.Dropout(0.5)),

                ("fc2", nn.Linear(4096, 4096, bias=bias)),
                ("fc2_bn", nn.BatchNorm1d(4096, affine=affine)),
                ("fc2_nonlin", nn.ReLU()),
                ("fc2_dropout", nn.Dropout(0.5)),

                ("fc3", nn.Linear(4096, num_classes, bias=bias)),
                ("fc3_bn", nn.BatchNorm1d(num_classes, affine=affine)),
            ]))

            self.sparse_layers = []
        elif model_type == "LeNet":
            self.model = nn.Sequential(OrderedDict([
                ("cnn1", nn.Conv2d(input_size[0], cnn_out_channels[0],
                                   kernel_size,)),
                ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0], affine=affine)),
                ("cnn1_nonlin", nn.ReLU()),
                ("cnn2", nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1],
                                   kernel_size)),
                ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[1], affine=affine)),
                ("cnn2_nonlin", nn.ReLU()),
                ("flatten", Flatten()),
                ("fc1", nn.Linear(
                    (feature_map_sidelength**2) * cnn_out_channels[1],
                    linear_units)),
                ("fc1_bn", nn.BatchNorm1d(linear_units, affine=affine)),
                ("fc1_nonlin", nn.ReLU()),
                ("fc1_dropout", Dropout()),
                ("fc2", nn.Linear(
                    linear_units, num_classes)),
            ]))
            self.sparse_layers = []
        elif model_type == "Binary" or model_type == "BinaryAlex":

            if config["local_reparam"]:
                Conv2d = bl.LocalReparamBinaryGatedConv2d  # noqa: N806
                Linear = bl.LocalReparamBinaryGatedLinear  # noqa: N806
            else:
                Conv2d = bl.BinaryGatedConv2d  # noqa: N806
                Linear = bl.BinaryGatedLinear  # noqa: N806

            if config["activation"] == "ReLU":
                NonlinearityFC = nn.ReLU  # noqa: N806
                NonlinearityConv1 = nn.ReLU  # noqa: N806
                NonlinearityConv2 = nn.ReLU  # noqa: N806
            elif config["activation"] == "KWinners":

                cnn_percent_on = (0.095, 0.125)
                linear_percent_on = 0.1

                nl_common = {
                    "boost_strength": 1.67,
                    "boost_strength_factor": 0.9,
                    "k_inference_factor": 1.5,
                    "duty_cycle_period": 1000,
                }

                NonlinearityFC = partial(  # noqa: N806
                    KWinners,
                    n=linear_units,
                    percent_on=linear_percent_on,
                    **nl_common
                )
                NonlinearityConv1 = partial(  # noqa: N806
                    KWinners2d,
                    channels=cnn_out_channels[0],
                    percent_on=cnn_percent_on[0],
                    local=False,
                    **nl_common
                )
                NonlinearityConv2 = partial(  # noqa: N806
                    KWinners2d,
                    channels=cnn_out_channels[1],
                    percent_on=cnn_percent_on[1],
                    local=False,
                    **nl_common
                )
            else:
                NonlinearityFC = partial(  # noqa: N806
                    bl.HeavisideStep,
                    neg1_activation=(config["activation"] == "BinaryNeg1"),
                    cancel_gradient=(config["cancel_gradient"]),
                    inplace=False)
                NonlinearityConv1 = NonlinearityFC  # noqa: N806
                NonlinearityConv2 = NonlinearityFC  # noqa: N806

            common_params = {
                "droprate_init": droprate_init,
                "l2_strength": l2_strength,
                "l0_strength": l0_strength,
                "learn_weight": learn_weight,
                "random_weight": random_weight,
                "bias": bias,
                "deterministic": deterministic,
                "one_sample_per_item": one_sample_per_item,
            }

            if model_type == "Binary":
                self.model = nn.Sequential(OrderedDict([
                    ("cnn1", Conv2d(input_size[0], cnn_out_channels[0],
                                    kernel_size, **common_params)),
                    ("cnn1_maxpool", nn.MaxPool2d(maxpool_stride)),
                    ("cnn1_bn", nn.BatchNorm2d(cnn_out_channels[0], affine=affine)),
                    ("cnn1_nonlin", NonlinearityConv1()),
                    ("cnn1_dropout", ConvDropout()),
                    ("cnn2", Conv2d(cnn_out_channels[0], cnn_out_channels[1],
                                    kernel_size, **common_params)),
                    ("cnn2_maxpool", nn.MaxPool2d(maxpool_stride)),
                    ("cnn2_bn", nn.BatchNorm2d(cnn_out_channels[1], affine=affine)),
                    ("cnn2_nonlin", NonlinearityConv2()),
                    ("cnn2_dropout", ConvDropout()),
                    ("flatten", Flatten()),
                    ("fc1", Linear(
                        (feature_map_sidelength**2) * cnn_out_channels[1],
                        linear_units, **common_params)),
                    ("fc1_bn", nn.BatchNorm1d(linear_units, affine=affine)),
                    ("fc1_nonlin", NonlinearityFC()),
                    ("fc1_dropout", Dropout()),
                    ("fc2", Linear(
                        linear_units, num_classes, **common_params)),
                ]))
                self.sparse_layers = ["cnn1", "cnn2", "fc1", "fc2"]
            elif model_type == "BinaryAlex":
                ratio_infl = config["ratio_infl"]

                self.model = nn.Sequential(OrderedDict([
                    # Don't use baseline bias on first layer!
                    ("cnn1", Conv2d(
                        input_size[0], 64 * ratio_infl, kernel_size, stride=2,
                        padding=1, **common_params)),
                    ("cnn1_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("cnn1_bn", nn.BatchNorm2d(64 * ratio_infl, affine=affine)),
                    ("cnn1_nonlin", Nonlinearity()),

                    ("cnn2", Conv2d(
                        64 * ratio_infl, 192 * ratio_infl, kernel_size=5,
                        padding=2, use_baseline_bias=baseline_bias,
                        **common_params)),
                    ("cnn2_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("cnn2_bn", nn.BatchNorm2d(192 * ratio_infl, affine=affine)),
                    ("cnn2_nonlin", Nonlinearity()),

                    ("cnn3", Conv2d(
                        192 * ratio_infl, 384 * ratio_infl, kernel_size=3,
                        padding=1, use_baseline_bias=baseline_bias,
                        **common_params)),
                    ("cnn3_bn", nn.BatchNorm2d(384 * ratio_infl, affine=affine)),
                    ("cnn3_nonlin", Nonlinearity()),

                    ("cnn4", Conv2d(
                        384 * ratio_infl, 256 * ratio_infl, kernel_size=3,
                        padding=1, use_baseline_bias=baseline_bias,
                        **common_params)),
                    ("cnn4_bn", nn.BatchNorm2d(256 * ratio_infl, affine=affine)),
                    ("cnn4_nonlin", Nonlinearity()),

                    ("cnn5", Conv2d(
                        256 * ratio_infl, 256, kernel_size=3, padding=1,
                        use_baseline_bias=baseline_bias, **common_params)),
                    ("cnn5_maxpool", nn.MaxPool2d(kernel_size=3, stride=2)),
                    ("cnn5_bn", nn.BatchNorm2d(256, affine=affine)),
                    ("cnn5_nonlin", Nonlinearity()),

                    ("flatten", Flatten()),

                    ("fc1", Linear(
                        256, 4096, use_baseline_bias=baseline_bias,
                        **common_params)),
                    ("fc1_bn", nn.BatchNorm1d(4096, affine=affine)),
                    ("fc1_nonlin", Nonlinearity()),
                    ("fc1_dropout", Dropout()),

                    ("fc2", Linear(
                        4096, 4096, use_baseline_bias=baseline_bias,
                        **common_params)),
                    ("fc2_bn", nn.BatchNorm1d(4096, affine=affine)),
                    ("fc2_nonlin", Nonlinearity()),
                    ("fc2_dropout", Dropout()),

                    ("fc3", Linear(
                        4096, num_classes, use_baseline_bias=baseline_bias,
                        **common_params)),
                    ("fc3_bn", nn.BatchNorm1d(num_classes, affine=affine)),
                ]))

                self.sparse_layers = ["cnn1", "cnn2", "cnn3", "cnn4", "cnn5",
                                      "fc1", "fc2", "fc3"]
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
        # pr = cProfile.Profile()
        # pr.enable()

        batch_size = (self.train_first_batch_size if self.iteration == 0
                      else self.train_batch_size)
        train_loader = torch.utils.data.DataLoader(
            preprocessed_dataset(
                DATAPATH / get_train_filename(self.iteration)),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        self.model.train()
        self.model.apply(update_boost_strength)

        if self.use_tqdm:
            batches = tqdm(train_loader, leave=False, desc="Training",
                           mininterval=self.tqdm_mininterval)
        else:
            batches = train_loader

        train_loss = 0.
        train_correct = 0.
        num_train_batches = 0

        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for data, target in batches:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_function(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                train_loss += loss.item()
                # print(loss)
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                num_train_batches += 1

            for layername in self.sparse_layers:
                layer = getattr(self.model, layername)
                layer.constrain_parameters()
                # print("layername",

        # prof.export_chrome_trace(os.path.expanduser("~/chrome-trace.trace"))

        if self.iteration != 0 and (self.iteration % self.lr_decay_period) == 0:
            self.scheduler.step()

        result = {
            "mean_train_accuracy": train_correct / len(train_loader.dataset),
            "mean_training_loss": train_loss / num_train_batches,
            "lr": self.optimizer.state_dict()["param_groups"][0]["lr"],
        }

        val_result = self.validate()
        result.update(val_result)

        if self.optimize_inference:
            for layername in self.sparse_layers:
                layer = getattr(self.model, layername)
                layer.optimize_inference = True
            val_opt_result = self.validate()
            result["mean_accuracy_optimized"] = val_opt_result["mean_accuracy"]
            for layername in self.sparse_layers:
                layer = getattr(self.model, layername)
                layer.optimize_inference = False

        result.update(self.nonzero_counts())

        expected_nz = 0
        inference_nz = 0
        flops = 0
        multiplies = 0
        for layername in self.sparse_layers:
            expected_nz += result[layername]["expected_nz"]
            inference_nz += result[layername]["inference_nz"]
            flops += result[layername]["flops"]
            multiplies += result[layername]["multiplies"]

        result["expected_nz"] = expected_nz
        result["inference_nz"] = inference_nz
        result["flops"] = flops
        result["multiplies"] = multiplies

        # pr.disable()
        # pstats.Stats(pr).dump_stats(os.path.expanduser("~/now{}.profile".format(self.iteration)))

        return result

    def validate(self):
        self.model.eval()
        val_loss = 0
        num_val_batches = 0
        val_correct = 0
        with torch.no_grad():
            if self.use_tqdm:
                batches = tqdm(self.val_loader, leave=False, desc="Testing")
            else:
                batches = self.val_loader

            for data, target in batches:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_function(output, target).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                num_val_batches += 1

        return {
            "mean_accuracy": val_correct / len(self.val_loader.dataset),
            "mean_loss": val_loss / num_val_batches,
            "total_correct": val_correct,
        }

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

    def loss_function(self, output, target):
        return self.loglike(output, target) + self.regularization()

    def regularization(self):
        reg = torch.tensor(0.).to(self.device)
        for layername in self.sparse_layers:
            layer = getattr(self.model, layername)
            reg += layer.regularization()
        return reg

    def nonzero_counts(self):
        result = {}

        for layername in self.sparse_layers:
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
                "hist_inference_nz_by_unit": i_nz_by_unit.tolist(),
                "inference_nz": torch.sum(i_nz_by_unit).item(),
                "num_input_units": num_inputs,
                "multiplies": multiplies,
                "flops": multiplies + adds,
            }

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HardConcrete",
                        choices=["HardConcrete", "Binary", "BinaryAlex",
                                 "Alex", "LeNet"])
    parser.add_argument("--activation", type=str, nargs="+",
                        default=["ReLU"],
                        choices=["ReLU", "BinaryNeg1", "Binary0",
                                 "Binary0Corrected", "KWinners"])
    parser.add_argument("--lr", type=float, nargs="+", default=[0.01])
    parser.add_argument("--lr-decay-period", type=int, nargs="+", default=[1])
    # Suggested: 7e-6 for HardConcrete, 1e-6 for Binary
    parser.add_argument("--l0", type=float, nargs="+", default=[7e-4])
    parser.add_argument("--l2", type=float, nargs="+", default=[0])
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.9825])
    # Suggested: 0.5 for HardConcrete, 0.8 for Binary
    parser.add_argument("--droprate-init", type=float, nargs="+", default=[0.5])
    parser.add_argument("--batch-size", type=int, nargs="+", default=[16])
    parser.add_argument("--first-batch-size", type=int, nargs="+", default=[16])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--ray-address", type=str, default="localhost:6379")
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--ratio-infl", type=int, default=3)
    parser.add_argument("--fixed-weight", action="store_true")
    parser.add_argument("--weight1", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cancel-gradient", action="store_true")
    parser.add_argument("--optimize-inference", action="store_true")
    parser.add_argument("--use-dropout", action="store_true")
    parser.add_argument("--one-sample-per-item", action="store_true")
    parser.add_argument("--local-reparam", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--checkpoint-at-end", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=0.1)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    # Verify that enough training data is present
    assert all((DATAPATH / get_train_filename(i)).exists()
               for i in range(args.epochs))

    if args.local:
        ray.init()
    else:
        ray.init(redis_address=args.ray_address)

    exp_name = "GSC-Permanence"
    print("Running experiment {}".format(exp_name))
    analysis = tune.run(StochasticGSCExperiment,
                        name=exp_name,
                        num_samples=args.samples,
                        checkpoint_freq=args.checkpoint_freq,
                        config={
                            "lr": tune.grid_search(args.lr),
                            # "l0_strength": tune.sample_from(
                            #     lambda _: math.exp(random.uniform(
                            #         math.log(5e-7), math.log(8e-5)))),
                            "l0_strength": tune.grid_search(args.l0),
                            "l2_strength": tune.grid_search(args.l2),
                            "model_type": tune.grid_search([args.model]),
                            "activation": tune.grid_search(args.activation),
                            "learn_weight": tune.grid_search([not args.fixed_weight]),
                            "random_weight": tune.grid_search([not args.weight1]),
                            "use_tqdm": args.progress,
                            "tqdm_mininterval": args.progress_interval,
                            "gamma": tune.grid_search(args.gamma),
                            "droprate_init": tune.grid_search(args.droprate_init),
                            "first_batch_size": tune.grid_search(args.first_batch_size),
                            "batch_size": tune.grid_search(args.batch_size),
                            "lr_decay_period": tune.grid_search(args.lr_decay_period),
                            "deterministic": tune.grid_search([args.deterministic]),
                            "cancel_gradient": tune.grid_search([args.cancel_gradient]),
                            "ratio_infl": tune.grid_search([args.ratio_infl]),
                            "optimize_inference": tune.grid_search(
                                [args.optimize_inference]),
                            "use_dropout": tune.grid_search([args.use_dropout]),
                            "one_sample_per_item": tune.grid_search(
                                [args.one_sample_per_item]),
                            "local_reparam": tune.grid_search([args.local_reparam]),
                        },
                        stop={"training_iteration": args.epochs},
                        checkpoint_at_end=args.checkpoint_at_end,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": (1 if torch.cuda.is_available() else 0)
                        },
                        loggers=(JsonLogger, CSVLogger, TFLoggerPlus),
                        verbose=1)

    print(("To browse results, instantiate "
           '`tune.Analysis("~/ray_results/{}")`').format(exp_name))
