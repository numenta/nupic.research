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

import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.image_transforms import RandomNoise
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.research.frameworks.pytorch.models import VGGSparseNet
from nupic.torch.modules import rezero_weights, update_boost_strength


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
        loaders.append(DataLoader(testset, batch_size=batch_size, shuffle=False,
                                  pin_memory=True))

    return loaders


class TinyCIFAR(object):
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
            train_dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True,
        )
        self.first_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=first_epoch_batch_size, shuffle=True,
            pin_memory=True,
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
            # or (epoch > 10 and ret['noise_accuracy'] < 0.40)
            # or (epoch > 30 and ret['noise_accuracy'] < 0.44)
            # or (epoch > 40 and ret['noise_accuracy'] < 0.50)
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

    def _create_vgg_model(self):
        self.model = VGGSparseNet(
            input_shape=(3, 32, 32),
            block_sizes=self.block_sizes,
            cnn_out_channels=self.cnn_out_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            cnn_weight_sparsity=self.cnn_weight_sparsity,
            cnn_percent_on=self.cnn_percent_on,
            linear_units=self.linear_n,
            linear_weight_sparsity=self.linear_weight_sparsity,
            linear_percent_on=self.linear_percent_on,
            k_inference_factor=self.k_inference_factor,
            boost_strength=self.boost_strength,
            boost_strength_factor=self.boost_strength_factor,
            use_max_pooling=self.use_max_pooling,
            num_classes=self.output_size
        )
        print(self.model)
        self.model.to(self.device)

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
        self.model.apply(rezero_weights)
        self.model.apply(update_boost_strength)
