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
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
import requests
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


def cnn_size(width, kernel_size, padding=1, stride=1):
    return (width - kernel_size + 2 * padding) / stride + 1


def create_test_loaders(noise_values, batch_size, data_dir):
    """Create a list of data loaders, one for each noise value."""
    print("Creating test loaders for noise values:", noise_values)
    loaders = []
    for noise in noise_values:

        transform_noise_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
                RandomNoise(noise, high_value=0.5 + 2 * 0.20, low_value=0.5 - 2 * 0.2),
            ]
        )

        testset = datasets.CIFAR10(
            root=data_dir, train=False, transform=transform_noise_test
        )
        loaders.append(DataLoader(testset, batch_size=batch_size, shuffle=False))

    return loaders


class TinyCIFARWeightInit(object):
    """Generic class for creating tiny CIFAR models. This can be used with Ray
    tune or PyExperimentSuite, to run a single trial or repetition of a
    network.

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
        """Tons of parameters!

        This should be called at the beginning of each repetition with a
        dict containing all the parameters required to setup the trial.
        """
        # Get trial parameters
        seed = config.get("seed", random.randint(0, 10000))
        self.data_dir = config["data_dir"]
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
        self.momentum = config["momentum"]
        self.weight_decay = config.get("weight_decay", 0.0005)
        self.learning_rate_gamma = config.get("learning_rate_gamma", 0.9)
        self.last_noise_results = None
        self.lr_step_schedule = config.get("lr_step_schedule", None)

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

        self.weight_init = config.get("weight_init", "default")

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
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            self.data_dir, train=True, transform=self.transform_train
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.first_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=first_epoch_batch_size, shuffle=True
        )
        self.test_loaders = create_test_loaders(
            self.noise_values, self.test_batch_size, self.data_dir
        )

        if network_type == "vgg":
            self._create_vgg_model()

        self.optimizer = self._create_optimizer(self.model)
        self.lr_scheduler = self._create_learning_rate_scheduler(self.optimizer)

    def train_epoch(self, epoch):
        """This should be called to do one epoch of training and testing.

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
        self._post_epoch(epoch)
        train_time = time.time() - t1

        ret = self.run_noise_tests(self.noise_values, self.test_loaders, epoch)

        # Hard coded early stopping criteria for quicker experimentation
        if (
            (epoch > 3 and abs(ret["mean_accuracy"] - 0.1) < 0.01)
            # or (ret['noise_accuracy'] > 0.66 and ret['test_accuracy'] > 0.91)
            or (ret["noise_accuracy"] > 0.69 and ret["test_accuracy"] > 0.91)
            or (ret["noise_accuracy"] > 0.62 and ret["test_accuracy"] > 0.92)
            # or (epoch > 10 and ret["noise_accuracy"] < 0.40)
            # or (epoch > 30 and ret["noise_accuracy"] < 0.44)
            # or (epoch > 40 and ret["noise_accuracy"] < 0.50)
        ):
            ret["stop"] = 1
        else:
            ret["stop"] = 0

        ret["epoch_time_train"] = train_time
        ret["epoch_time"] = time.time() - t1
        ret["learning_rate"] = self.learning_rate
        # print(epoch, ret)
        return ret

    def model_save(self, checkpoint_dir):
        """Save the model in this directory.

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
        """Add a single CNN layer to our modules."""
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

    def _create_optimizer(self, model):
        """Create a new instance of the optimizer."""
        return torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def _create_learning_rate_scheduler(self, optimizer):
        """Creates the learning rate scheduler and attach the optimizer."""
        if self.lr_step_schedule is not None:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=self.learning_rate_gamma
            )
        else:
            return None

    def _adjust_learning_rate(self, optimizer, epoch):
        if self.lr_step_schedule is not None:
            if epoch in self.lr_step_schedule:
                self.learning_rate *= self.learning_rate_gamma
                print("Reducing learning rate to:", self.learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.learning_rate
        else:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                self.learning_rate = self.lr_scheduler.get_lr()[0]
                print("Reducing learning rate to:", self.learning_rate)

    def run_noise_tests(self, noise_values, loaders, epoch):
        """
        Test the model with different noise values and return test metrics.
        """
        ret = self.last_noise_results

        # Just do noise tests every 3 iterations, about a 2X overall speedup
        if epoch % 3 == 0 or ret is None:
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

    def _post_epoch(self, epoch):
        """
        The set of actions to do after each epoch of training: adjust learning
        rate, rezero sparse weights, and update boost strengths.
        """
        self._adjust_learning_rate(self.optimizer, epoch)
        self.model.apply(update_boost_strength)

    def _post_batch(self, *args, **kwargs):
        """
        The set of actions to do after each batch of training: rezero sparse
        weights
        """
        self.model.apply(rezero_weights)

    def _initialize_weights(self):
        if self.weight_init == "lsuv":
            initializer = LSUVWeightInit(
                self.model, self.train_loader, device=self.device
            )
            initializer.initialize()
        elif self.weight_init == "grassmannian":
            initializer = GrassmannianWeightInit(self.model, device=self.device)
            initializer.initialize()
        elif self.weight_init == "default":
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


class GrassmannianWeightInit(object):
    """
    Set first conv layer weights to Grassmannian as per Prabhu and Yap 2019
    From the paper by J. H. Conway, R. H. Hardin and N. J. A. Sloane
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device

    def grassmannian_extract(self, n, k_size, num_channels):
        """
        Download packed subspace from Sloane (http://neilsloane.com/grass/)
        """
        target_url = "http://neilsloane.com/grass/dim{}/grassc.{}.{}.{}.txt".format(
            k_size, k_size, num_channels, n
        )
        response = requests.get(target_url)
        list_str = response.text.split("\n")[0 : num_channels * k_size * n]
        k = int(np.sqrt(k_size))
        w_mat = np.float_(list_str).reshape(n, k, k, num_channels)
        print("Downloaded weights of shape: %s" % str(w_mat.shape))
        return w_mat

    def initialize(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                n_filters = m.out_channels
                n_input = m.in_channels
                k = m.kernel_size[0]
                w = self.grassmannian_extract(n_filters, k * k, n_input)
                m.weight.data = torch.from_numpy(w).float().to(self.device)
                # Break after first Conv layer
                break


class LSUVWeightInit(object):
    """
    Layer-sequential unit variance (LSUV) initialization from Mishkin and Matas 2016
    https://arxiv.org/abs/1511.06422
    """

    def __init__(self, model, data_loader, device=None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        # Holder for parameters needed for LSUV init
        self.lsuv_data = {
            "act_dict": None,  # Output
            "hook": None,  # Forward hook,
            "current_coef": None,  # Mult for weights
            "layers_done": -1,
            "hook_idx": 0,
            "correction_counter": 0,
            "correction_needed": False,
            "n_layers": 0,
        }
        print("Starting LSUV weight init...")

    def count_conv_fc(self, m):
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, SparseWeights2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, SparseWeights)
        ):
            self.lsuv_data["n_layers"] += 1

    def store_activations(self, module, _input, _output):
        # Store output of this layer on each forward pass
        self.lsuv_data["act_dict"] = _output.data.cpu().numpy()

    def add_hook(self, m):
        """
        Add forward hook to each layer
        """
        if self.lsuv_data["hook_idx"] > self.lsuv_data["layers_done"]:
            self.lsuv_data["hook"] = m.register_forward_hook(self.store_activations)
        else:
            # Done, skip
            self.lsuv_data["hook_idx"] += 1

    def update_weights(self, m):
        if self.lsuv_data["hook"] is None:
            return
        if not self.lsuv_data["correction_needed"]:
            return
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, SparseWeights2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, SparseWeights)
        ):
            if self.lsuv_data["correction_counter"] < self.lsuv_data["hook_idx"]:
                self.lsuv_data["correction_counter"] += 1
            else:
                m.weight.data *= self.lsuv_data["current_coef"]
                self.lsuv_data["correction_needed"] = False

    def orthogonal_weight_init(self, m):
        # Fill with semi-orthogonal matrix as per Saxe et al 2013
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)

    def initialize(self, tol_var=0.1, max_attempts=30):
        """
        Updates weights in model
        """
        print("Running LSUV weight initialization...")

        self.model.eval()
        self.model.apply(self.orthogonal_weight_init)
        data_iter = iter(self.data_loader)
        self.model.apply(self.count_conv_fc)

        for _idx in range(self.lsuv_data["n_layers"]):
            self.model.apply(self.add_hook)
            data, target = next(data_iter)
            data, target = data.to(self.device), target.to(self.device)
            self.model(data)
            attempts = 0
            current_sd = self.lsuv_data.get("act_dict").std()
            while abs(current_sd - 1.0) >= tol_var and (attempts < max_attempts):
                self.lsuv_data["current_coef"] = 1.0 / (current_sd + 1e-8)
                self.lsuv_data["correction_needed"] = True

                self.model.apply(self.update_weights)

                data, target = next(data_iter)
                data, target = data.to(self.device), target.to(self.device)
                self.model(data)
                current_sd = self.lsuv_data.get("act_dict").std()  # Repeated code?
                attempts += 1
            if attempts == max_attempts:
                print(
                    "Failed to converge after %d attempts, sd: %.3f"
                    % (attempts, current_sd)
                )
            else:
                print("Converged after %d attempts, sd: %.3f" % (attempts, current_sd))

            # Remove forward hook
            if self.lsuv_data["hook"] is not None:
                self.lsuv_data["hook"].remove()
            self.lsuv_data["hook"] = None
            self.lsuv_data["layers_done"] += 1
            self.lsuv_data["hook_idx"] = 0
            self.lsuv_data["correction_counter"] = 0
