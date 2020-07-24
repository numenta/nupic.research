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
# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

import math
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.image_transforms import RandomNoise
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.torch.modules import (
    Flatten,
    KWinners,
    KWinners2d,
    SparseWeights,
    SparseWeights2d,
    rezero_weights,
    update_boost_strength,
)


# monkey patch ReduceLROnPlateau, no get_lr implemented
def get_lr(self):
    for param_group in self.optimizer.param_groups:
        return param_group["lr"]


torch.optim.lr_scheduler.ReduceLROnPlateau.get_lr = get_lr


def cnn_size(width, kernel_size, padding=1, stride=1):
    return (width - kernel_size + 2 * padding) / stride + 1


def create_test_loaders(dataset, noise_values, batch_size, data_dir):
    """
    Create a list of data loaders, one for each noise value
    """
    print("Creating test loaders for noise values:", noise_values)
    loaders = []
    for noise in noise_values:

        transform_noise_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.50707516, 0.48654887, 0.44091784),
                    (0.26733429, 0.25643846, 0.27615047),
                ),
                RandomNoise(noise, high_value=0.5 + 2 * 0.20, low_value=0.5 - 2 * 0.2),
            ]
        )

        testset = getattr(datasets, dataset)(
            root=data_dir, train=False, transform=transform_noise_test
        )
        loaders.append(DataLoader(testset, batch_size=batch_size, shuffle=False))

    return loaders


class TinyCIFAR(object):
    """
    Generic class for creating tiny CIFAR models. This can be used with Ray tune
    or PyExperimentSuite, to run a single trial or repetition of a network.

    The correct way to use this from the outside is:

    model = TinyCIFAR()
    model.model_setup(config_dict)

    for epoch in range(10):
      model.train_epoch(epoch)
    model.model_save(path)

    new_model = TinyCIFAR()
    new_model.model_restore(path)
    """

    def __init__(self):
        pass

    def model_setup(self, config):
        """
        Tons of parameters!

        This should be called at the beginning of each repetition with a dict
        containing all the parameters required to setup the trial.
        """
        # Get trial parameters
        seed = config.get("seed", random.randint(0, 10000))
        self.data_dir = os.path.expanduser(config.get("data_dir", "data"))
        self.model_filename = config.get("model_filename", "model.pth")
        self.iterations = config["iterations"]

        # Training / testing parameters
        batch_size = config["batch_size"]
        first_epoch_batch_size = config.get("first_epoch_batch_size", batch_size)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_first_epoch = config.get(
            "batches_in_first_epoch", self.batches_in_epoch
        )

        self.test_batch_size = config["test_batch_size"]
        self.test_batches_in_epoch = config.get("test_batches_in_epoch", sys.maxsize)
        self.noise_values = config.get("noise_values", [0.0, 0.1])
        self.loss_function = nn.functional.cross_entropy

        self.learning_rate = config["learning_rate"]
        self.momentum = config.get("momentum", 0.5)
        self.weight_decay = config.get("weight_decay", 0.0005)
        self.learning_rate_gamma = config.get("learning_rate_gamma", 0.9)
        self.last_noise_results = None
        self.lr_step_schedule = config.get("lr_step_schedule", None)
        self.early_stopping = config.get("early_stopping", None)

        # Network parameters
        network_type = config.get("network_type", "vgg")
        in_channels, self.h, self.w = config["input_shape"]

        self.boost_strength = config["boost_strength"]
        self.boost_strength_factor = config["boost_strength_factor"]
        self.k_inference_factor = config["k_inference_factor"]

        # CNN parameters - these are lists, one for each CNN layer
        self.cnn_percent_on = config["cnn_percent_on"]
        self.cnn_kernel_sizes = config.get(
            "cnn_kernel_size", [3] * len(self.cnn_percent_on)
        )
        self.cnn_out_channels = config.get(
            "cnn_out_channels", [32] * len(self.cnn_percent_on)
        )
        self.cnn_weight_sparsity = config.get(
            "cnn_weight_sparsity", [1.0] * len(self.cnn_percent_on)
        )
        self.in_channels = [in_channels] + self.cnn_out_channels
        self.block_sizes = config.get("block_sizes", [1] * len(self.cnn_percent_on))
        self.use_max_pooling = config.get("use_max_pooling", False)

        # Linear parameters
        self.linear_weight_sparsity = config["weight_sparsity"]
        self.linear_n = config["linear_n"]
        self.linear_percent_on = config["linear_percent_on"]
        if isinstance(self.linear_n, int):
            self.linear_n = [self.linear_n]
            self.linear_percent_on = [self.linear_percent_on]
            self.linear_weight_sparsity = [self.linear_weight_sparsity]
        self.output_size = config.get("output_size", 10)
        self.optimizer_alg = config.get("optimizer", "SGD")

        # Setup devices, model, and dataloaders
        print("setup: Torch device count=", torch.cuda.device_count())
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            print("setup: Using cuda")
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(seed)
        else:
            print("setup: Using cpu")
            self.device = torch.device("cpu")

        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # can store stats in database or dynamically obtain
                transforms.Normalize(
                    (0.50707516, 0.48654887, 0.44091784),
                    (0.26733429, 0.25643846, 0.27615047),
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.50707516, 0.48654887, 0.44091784),
                    (0.26733429, 0.25643846, 0.27615047),
                ),
            ]
        )

        # added custom dataset and output sizes to reuse model
        self.output_size = config.get("output_size", 10)
        self.dataset = config.get("dataset", "CIFAR10")

        train_dataset = getattr(datasets, self.dataset)(
            self.data_dir, train=True, transform=self.transform_train
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.first_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=first_epoch_batch_size, shuffle=True
        )
        self.test_loaders = create_test_loaders(
            self.dataset, self.noise_values, self.test_batch_size, self.data_dir
        )

        if network_type == "vgg":
            self._create_vgg_model()

        self.optimizer = self._create_optimizer(self.model, self.optimizer_alg)
        self.lr_scheduler = self._create_learning_rate_scheduler(self.optimizer)

        # adding track of losses for early stopping
        # self.mean_losses = deque(maxlen=max(3,int(self.iterations/10)))
        self.mean_losses = deque(maxlen=self.iterations)
        self.bad_epochs = 0
        self.grace_period = max(1, int(self.iterations / 5))
        self.patience = 3

    def train_epoch(self, epoch):
        """
        This should be called to do one epoch of training and testing.

        Returns:
        A dict that describes progress of this epoch.
        The dict includes the key 'stop'. If set to one, this network
        should be stopped early. Training is not progressing well enough.
        """
        t1 = time.time()
        if epoch == 0:
            train_loader = self.first_loader
            batches_in_epoch = self.batches_in_first_epoch
        else:
            train_loader = self.train_loader
            batches_in_epoch = self.batches_in_epoch

        train_model(
            model=self.model,
            loader=train_loader,
            optimizer=self.optimizer,
            device=self.device,
            batches_in_epoch=batches_in_epoch,
            criterion=self.loss_function,
            post_batch_callback=self._post_batch,
        )

        train_time = time.time() - t1

        ret = self.run_noise_tests(self.noise_values, self.test_loaders, epoch)
        self._post_epoch(epoch, ret["mean_loss"])

        if self.early_stopping:
            ret["stop"] = self._early_stopping(epoch, ret["mean_loss"])
        else:
            ret["stop"] = 0
        ret["epoch_time_train"] = train_time
        ret["epoch_time"] = time.time() - t1
        ret["learning_rate"] = self.learning_rate
        # print(epoch, ret)
        return ret

    def model_save(self, checkpoint_dir):
        """
        Save the model in this directory.
        :param checkpoint_dir:

        :return: str: The return value is expected to be the checkpoint path that
        can be later passed to `model_restore()`.
        """
        # checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        # torch.save(self.model.state_dict(), checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_dir, self.model_filename)

        # Use the slow method if filename ends with .pt
        if checkpoint_path.endswith(".pt"):
            torch.save(self.model, checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)

        return checkpoint_path

    def model_restore(self, checkpoint_path):
        """
        :param checkpoint_path: Loads model from this checkpoint path.
        If path is a directory, will append the parameter model_filename

        """
        print("loading from", checkpoint_path)
        if os.path.isdir(checkpoint_path):
            checkpoint_file = os.path.join(checkpoint_path, self.model_filename)
        else:
            checkpoint_file = checkpoint_path

        # Use the slow method if filename ends with .pt
        if checkpoint_file.endswith(".pt"):
            self.model = torch.load(checkpoint_file, map_location=self.device)
        else:
            self.model.load_state_dict(
                torch.load(checkpoint_file, map_location=self.device)
            )

    def _add_cnn_layer(
        self,
        index_str,
        in_channels,
        out_channels,
        kernel_size,
        percent_on,
        weight_sparsity,
        add_pooling,
    ):
        """
        Add a single CNN layer to our modules
        """
        # Add CNN layer
        if kernel_size == 3:
            padding = 1
        else:
            padding = 2

        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        if weight_sparsity < 1.0:
            conv2d = SparseWeights2d(conv2d, weight_sparsity=weight_sparsity)
        self.model.add_module("cnn_" + index_str, conv2d)

        self.model.add_module("bn_" + index_str, nn.BatchNorm2d(out_channels)),

        if add_pooling:
            if self.use_max_pooling:
                self.model.add_module(
                    "maxpool_" + index_str, nn.MaxPool2d(kernel_size=2, stride=2)
                )
            else:
                self.model.add_module(
                    "avgpool_" + index_str, nn.AvgPool2d(kernel_size=2, stride=2)
                )

        if percent_on < 1.0:
            self.model.add_module(
                "kwinners_2d_" + index_str,
                KWinners2d(
                    percent_on=percent_on,
                    channels=out_channels,
                    k_inference_factor=self.k_inference_factor,
                    boost_strength=self.boost_strength,
                    boost_strength_factor=self.boost_strength_factor,
                ),
            )
        else:
            self.model.add_module("ReLU_" + index_str, nn.ReLU(inplace=True))

    def _create_vgg_model(self):
        """
        block_sizes = [1,1,1] - number of CNN layers in each block
        cnn_out_channels = [c1, c2, c3] - # out_channels in each layer of this block
        cnn_kernel_size = [k1, k2, k3] - kernel_size in each layer of this block
        cnn_weight_sparsity = [w1, w2, w3] - weight sparsity of each layer of this block
        cnn_percent_on = [p1, p2, p3] - percent_on in each layer of this block
        """
        # Here we require exactly 3 blocks
        # assert(len(self.block_sizes) == 3)

        # Create simple CNN model, with options for sparsity
        self.model = nn.Sequential()

        in_channels = 3
        output_size = 32 * 32
        output_units = output_size * in_channels
        for l, block_size in enumerate(self.block_sizes):
            for b in range(block_size):
                self._add_cnn_layer(
                    index_str=str(l) + "_" + str(b),
                    in_channels=in_channels,
                    out_channels=self.cnn_out_channels[l],
                    kernel_size=self.cnn_kernel_sizes[l],
                    percent_on=self.cnn_percent_on[l],
                    weight_sparsity=self.cnn_weight_sparsity[l],
                    add_pooling=b == block_size - 1,
                )
                in_channels = self.cnn_out_channels[l]
            output_size = int(output_size / 4)
            output_units = output_size * in_channels

        # Flatten CNN output before passing to linear layer
        self.model.add_module("flatten", Flatten())

        # Linear layer
        input_size = output_units
        for l, linear_n in enumerate(self.linear_n):
            linear = nn.Linear(input_size, linear_n)
            if self.linear_weight_sparsity[l] < 1.0:
                self.model.add_module(
                    "linear_" + str(l),
                    SparseWeights(linear, self.linear_weight_sparsity[l]),
                )
            else:
                self.model.add_module("linear_" + str(l), linear)

            if self.linear_percent_on[l] < 1.0:
                self.model.add_module(
                    "kwinners_linear_" + str(l),
                    KWinners(
                        n=linear_n,
                        percent_on=self.linear_percent_on[l],
                        k_inference_factor=self.k_inference_factor,
                        boost_strength=self.boost_strength,
                        boost_strength_factor=self.boost_strength_factor,
                    ),
                )
            else:
                self.model.add_module("Linear_ReLU_" + str(l), nn.ReLU())

            input_size = self.linear_n[l]

        # Output layer
        self.model.add_module("output", nn.Linear(input_size, self.output_size))

        print(self.model)

        self.model.to(self.device)

        self._initialize_weights()

    def _create_optimizer(self, model, optimizer="Adam"):
        """
        Create a new instance of the optimizer
        """
        if optimizer == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif optimizer == "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError("{} is not a valid optimizer".format(optimizer))

    def _create_learning_rate_scheduler(self, optimizer, scheduler="ReduceLROnPlateau"):
        """
        Creates the learning rate scheduler and attach the optimizer
        If step schedule is a list, don't create a scheduler
        """
        if self.lr_step_schedule and not isinstance(self.lr_step_schedule, list):
            if scheduler == "StepLR":
                return torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=1, gamma=self.learning_rate_gamma
                )
            elif scheduler == "ReduceLROnPlateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",  # loss
                    patience=5,
                    threshold=1e-2,
                    factor=self.learning_rate_gamma,
                )
            else:
                raise ValueError(
                    "{} is not a valid learning rate scheduler".format(scheduler)
                )
        else:
            return None

    def _adjust_learning_rate(self, optimizer, epoch, metric):
        """Accepts a schedule either as a list of steps or a boolean"""
        if self.lr_step_schedule and isinstance(self.lr_step_schedule, list):
            if epoch in self.lr_step_schedule:
                self.learning_rate *= self.learning_rate_gamma
                print("Reducing learning rate to:", self.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
        else:
            if self.lr_scheduler:
                self.lr_scheduler.step(metric)
                self.learning_rate = np.mean(self.lr_scheduler.get_lr())

    def _early_stopping(self, epoch, metric):
        """Custom early stopping based on moving median"""
        self.mean_losses.append(metric)
        if metric >= np.median(self.mean_losses):
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        if epoch > self.grace_period:
            if self.bad_epochs > self.patience:
                return 1

        return 0

    def run_noise_tests(self, noise_values, loaders, epoch):
        """
        Test the model with different noise values and return test metrics.
        """
        ret = self.last_noise_results

        # Just do noise tests every 3 iterations, about a 2X overall speedup
        # back to 1 temporarily
        if epoch % 1 == 0 or ret is None:
            ret = {"noise_values": noise_values, "noise_accuracies": []}
            accuracy = 0.0
            loss = 0.0
            for _noise, loader in zip(noise_values, loaders):
                test_result = evaluate_model(
                    model=self.model,
                    loader=loader,
                    device=self.device,
                    batches_in_epoch=self.test_batches_in_epoch,
                    criterion=self.loss_function,
                )
                accuracy += test_result["mean_accuracy"]
                loss += test_result["mean_loss"]
                ret["noise_accuracies"].append(test_result["mean_accuracy"])

            ret["mean_accuracy"] = accuracy / len(noise_values)
            ret["test_accuracy"] = ret["noise_accuracies"][0]
            ret["noise_accuracy"] = ret["noise_accuracies"][-1]
            ret["mean_loss"] = loss / len(noise_values)

            self.last_noise_results = ret

        return ret

    def _post_batch(self, *args, **kwargs):
        self.model.apply(rezero_weights)

    def _post_epoch(self, epoch, metric):
        """
        The set of actions to do after each epoch of training:
        1.adjust learning rate,
        2.rezero sparse weights,
        3. and update boost strengths.
        """
        self._adjust_learning_rate(self.optimizer, epoch, metric)
        self.model.apply(update_boost_strength)

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
