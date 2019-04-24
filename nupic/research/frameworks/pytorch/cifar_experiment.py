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

import time

import torch
import torch.nn as nn
from nupic.torch.modules import (
  SparseWeights, SparseWeights2d,
  Flatten, KWinners2d, KWinners,
  rezeroWeights, updateBoostStrength)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.model_utils import *
from nupic.research.frameworks.pytorch.image_transforms import *


def CNNSize(width, kernel_size, padding=1, stride=1):
  return (width - kernel_size + 2 * padding) / stride + 1


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
    This should be called at the beginning of each repetition with a dict
    containing all the parameters required to setup the trial.
    """
    # Get trial parameters
    seed = config["seed"]
    self.data_dir = config["data_dir"]
    self.model_filename = config.get("model_filename", "model.pt")

    # Training / testing parameters
    batch_size = config["batch_size"]
    first_epoch_batch_size = config.get("first_epoch_batch_size", batch_size)
    self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
    self.batches_in_first_epoch = config.get("batches_in_first_epoch",
                                             self.batches_in_epoch)

    self.test_batch_size = config["test_batch_size"]
    self.test_batches_in_epoch = config.get("test_batches_in_epoch", sys.maxsize)
    self.noise_values = config.get("noise_values", [0.0, 0.1])
    self.loss_function = nn.functional.cross_entropy

    self.learning_rate = config["learning_rate"]
    self.momentum = config["momentum"]
    self.weight_decay = config.get("weight_decay", 0.0005)
    self.learning_rate_gamma = config.get("learning_rate_gamma", 0.9)

    # Network parameters
    network_type = config["network_type"]
    inChannels, self.h, self.w = config["input_shape"]

    self.boost_strength = config["boost_strength"]
    self.boost_strength_factor = config["boost_strength_factor"]
    self.k_inference_factor = config["k_inference_factor"]

    # CNN parameters - these are lists, one for each CNN layer
    self.cnn_k = config["cnn_k"]
    self.cnn_kernel_sizes = config.get("cnn_kernel_size", [3] * len(self.cnn_k))
    self.cnn_out_channels = config.get("cnn_out_channels", [32] * len(self.cnn_k))
    self.cnn_weight_sparsity = config.get("cnn_weight_sparsity",
                                          [1.0]*len(self.cnn_k))
    self.in_channels = [inChannels] + self.cnn_out_channels

    # Linear parameters
    self.weight_sparsity = config["weight_sparsity"]
    self.linear_n = config["linear_n"]
    self.linear_k = config["linear_k"]
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

    self.transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    self.transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(self.data_dir, train=True,
                                     transform=self.transform_train)

    self.train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True
    )
    self.first_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=first_epoch_batch_size, shuffle=True
    )
    self.test_loaders = self._createTestLoaders(self.noise_values)

    if network_type == "tiny_sparse":
      self._createTinySparseModel()

    self.optimizer = self._createOptimizer(self.model)
    self.lr_scheduler = self._createLearningRateScheduler(self.optimizer)


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

    self._preEpoch()
    trainModel(model=self.model, loader=train_loader,
               optimizer=self.optimizer, device=self.device,
               batches_in_epoch=batches_in_epoch,
               criterion=self.loss_function)
    self._postEpoch()
    trainTime = time.time() - t1

    ret = self._runNoiseTests(self.noise_values, self.test_loaders)

    # Early stopping criterion
    if epoch > 1 and abs(ret['mean_accuracy'] - 0.1) < 0.01:
      ret['stop'] = 1
    else:
      ret['stop'] = 0

    ret['epoch_time_train'] = trainTime
    ret['epoch_time'] = time.time() - t1
    ret["learning_rate"] = self.lr_scheduler.get_lr()[0]
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
    torch.save(self.model, checkpoint_path)
    return checkpoint_path


  def model_restore(self, checkpoint_path):
    """
    :param checkpoint_path: Loads model from this checkpoint path
    :return:
    """
    print("loading from", checkpoint_path)
    self.model = torch.load(checkpoint_path, map_location=self.device)
    # self.model.load_state_dict(checkpoint_path)


  def _createTestLoaders(self, noise_values, batch_size=None):
    """
    Create a list of data loaders, one for each noise value
    """
    print("Creating test loaders for noise values:", noise_values)
    loaders = []
    for noise in noise_values:

      transform_noise_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        RandomNoise(noise,
                    highValue=0.5 + 2 * 0.20,
                    lowValue=0.5 - 2 * 0.2,
                    ),
      ])

      testset = datasets.CIFAR10(root=self.data_dir,
                                 train=False,
                                 transform=transform_noise_test)
      loaders.append(
        DataLoader(testset,
                   batch_size=batch_size or self.test_batch_size,
                   shuffle=False)
      )

    return loaders


  def _createTinySparseModel(self):
    prev_w = self.w
    cnn_output_len = []
    padding = []
    for i, (kernel_size, ch) in enumerate(zip(self.cnn_kernel_sizes,
                                              self.cnn_out_channels)):
      if kernel_size == 3:
        padding.append(1)
      else:
        padding.append(2)
      cnn_w = CNNSize(prev_w, kernel_size, padding=padding[-1]) // 2
      cnn_output_len.append(int(ch * (cnn_w) ** 2))
      prev_w = cnn_w


    # Create simple sparse model
    self.model = nn.Sequential()

    for c, l in enumerate(cnn_output_len):

      # Sparse CNN layer
      conv2d = nn.Conv2d(in_channels=self.in_channels[c],
                         out_channels=self.cnn_out_channels[c],
                         kernel_size=self.cnn_kernel_sizes[c],
                         padding=padding[c], bias=False)
      if self.cnn_weight_sparsity[c] < 1.0:
        conv2d = SparseWeights2d(conv2d,
                                 weightSparsity=self.cnn_weight_sparsity[c])
      self.model.add_module("cnn_"+str(c), conv2d)

      # Batch norm plus average pooling
      self.model.add_module("bn_" + str(c),
                            nn.BatchNorm2d(self.cnn_out_channels[c])),
      self.model.add_module("avgpool_"+str(c), nn.AvgPool2d(kernel_size=2))

      # K-winners, if required
      if self.cnn_k[c] < 1.0:
        self.model.add_module("kwinners_2d_"+str(c),
                              KWinners2d(n=cnn_output_len[c],
                                         channels=self.cnn_out_channels[c],
                                         k=int(self.cnn_k[c] * cnn_output_len[c]),
                                         kInferenceFactor=self.k_inference_factor,
                                         boostStrength=self.boost_strength,
                                         boostStrengthFactor=self.boost_strength_factor))


    # Flatten CNN output before passing to linear layer
    self.model.add_module("flatten", Flatten())

    # Linear layer
    linear = nn.Linear(cnn_output_len[-1], self.linear_n)
    if self.weight_sparsity < 1.0:
      self.model.add_module("linear",
                            SparseWeights(linear, self.weight_sparsity))
    else:
      self.model.add_module("linear", linear)

    if self.linear_k < 1.0:
      self.model.add_module("kwinners_linear",
                          KWinners(n=self.linear_n,
                                   k=int(self.linear_k * self.linear_n),
                                   kInferenceFactor=self.k_inference_factor,
                                   boostStrength=self.boost_strength,
                                   boostStrengthFactor=self.boost_strength_factor))

    # Output layer
    self.model.add_module("output", nn.Linear(self.linear_n, self.output_size))

    print(self.model)

    self.model.to(self.device)


  def _createOptimizer(self, model):
    """
    Create a new instance of the optimizer
    """
    return torch.optim.SGD(model.parameters(),
                           lr=self.learning_rate,
                           momentum=self.momentum,
                           weight_decay=self.weight_decay)


  def _createLearningRateScheduler(self, optimizer):
    """
    Creates the learning rate scheduler and attach the optimizer
    """
    return torch.optim.lr_scheduler.StepLR(optimizer,
                                           step_size=1,
                                           gamma=self.learning_rate_gamma)


  def _runNoiseTests(self, noiseValues, loaders):
    """
    Test the model with different noise values and return test metrics.
    """
    ret = {
      'noise_values': noiseValues,
      'noise_accuracies': [],
    }
    accuracy = 0.0
    loss = 0.0
    for noise, loader in zip(noiseValues, loaders):
      testResult = evaluateModel(
        model=self.model,
        loader=loader,
        device=self.device,
        batches_in_epoch=self.test_batches_in_epoch,
        criterion=self.loss_function
      )
      accuracy += testResult['mean_accuracy']
      loss += testResult['mean_loss']
      ret['noise_accuracies'].append(testResult['mean_accuracy'])

    ret['mean_accuracy'] = accuracy / len(noiseValues)
    ret['noise_accuracy'] = ret['noise_accuracies'][-1]
    ret['mean_loss'] = loss / len(noiseValues)
    return ret


  def _preEpoch(self):
    if self.lr_scheduler is not None:
      self.lr_scheduler.step()


  def _postEpoch(self):
    self.model.apply(rezeroWeights)
    self.model.apply(updateBoostStrength)

