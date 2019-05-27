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

import logging
import time

import torch
import torch.backends
import torch.cuda
import torch.nn as nn
import torch.optim
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.image_transforms import *
from nupic.research.frameworks.pytorch.model_utils import (
  trainModel, evaluateModel, setRandomSeed)



def getLogger(name, verbose):
  """
  Configure Logger based on verbose level (0: ERROR, 1: INFO, 2: DEBUG)
  """
  logger = logging.getLogger(name)
  if verbose == 0:
    logger.setLevel(logging.ERROR)
  elif verbose == 1:
    logger.setLevel(logging.INFO)
  else:
    logger.setLevel(logging.DEBUG)

  return logger



class MobileNetCIFAR10(object):
  """
  CIFAR-10 experiment using MobileNet
  """


  def __init__(self, config):
    super(MobileNetCIFAR10, self).__init__()

    self.logger = getLogger(config["name"], config["verbose"])
    self.logger.debug("Config: %s", config)

    # Setup random seed
    seed = config["seed"]
    setRandomSeed(seed)

    self._configureDataloaders(config)

    # Configure Model
    model_type = config["model_type"]
    model_params = config["model_params"]
    self.model = model_type(**model_params)
    self.logger.debug("Model: %s", self.model)

    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      self.model = self.model.cuda()
    else:
      self.device = torch.device("cpu")

    # Configure Optimizer. Skip weight decay on deep-wise
    params = [{"params": self.model.conv.parameters()},
              {"params": self.model.deepwise.parameters(), "weight_decay": 0},
              {"params": self.model.classifier.parameters()}]

    self.optimizer = torch.optim.RMSprop(params, lr=config["learning_rate"],
                                         weight_decay=config["weight_decay"])
    self.loss_function = config["loss_function"]
    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=config["lr_step_size"],
                                                        gamma=config["learning_rate_gamma"])
    if torch.cuda.device_count() > 1:
      self.model = torch.nn.DataParallel(self.model)

    self.batches_in_epoch = config["batches_in_epoch"]
    self.batches_in_first_epoch = config["batches_in_first_epoch"]
    self.test_batches_in_epoch = config["test_batches_in_epoch"]

    self.config = config


  def train(self, epoch):
    if epoch == 0:
      train_loader = self.first_loader
      batches_in_epoch = self.batches_in_first_epoch
    else:
      train_loader = self.train_loader
      batches_in_epoch = self.batches_in_epoch

    self.logger.info("epoch: %s", epoch)
    t0 = time.time()

    self.preEpoch()
    trainModel(model=self.model, loader=train_loader,
               optimizer=self.optimizer, device=self.device,
               batches_in_epoch=batches_in_epoch,
               criterion=self.loss_function)
    self.postEpoch()

    self.logger.info("training duration: %s", time.time() - t0)


  def test(self, loader=None):
    t0 = time.time()
    if loader is None:
      loader = self.test_loader
    results = evaluateModel(model=self.model, loader=loader,
                            device=self.device,
                            batches_in_epoch=self.test_batches_in_epoch,
                            criterion=self.loss_function)
    self.logger.info("testing duration: %s", time.time() - t0)
    self.logger.info("mean_accuracy: %s", results["mean_accuracy"])
    self.logger.info("mean_loss: %s", results["mean_loss"])
    return results


  def save(self, checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, "model.pt")
    torch.save(self.model.state_dict(), checkpoint_path)
    return checkpoint_path


  def restore(self, checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, "model.pt")
    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


  def preEpoch(self):
    pass


  def postEpoch(self):
    self.lr_scheduler.step()
    self.logger.info("learning rate: %s", self.lr_scheduler.get_lr())


  def _configureDataloaders(self, config):
    data_dir = config["data_dir"]
    batch_size = config["batch_size"]
    test_batch_size = config["test_batch_size"]
    first_epoch_batch_size = config["first_epoch_batch_size"]

    transform = [
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]

    # Add noise
    noise = config.get("noise", 0.0)
    if noise > 0.0:
      transform.append(RandomNoise(noise,
                                   highValue=0.5 + 2 * 0.2,
                                   lowValue=0.5 - 2 * 0.2))

    train_dataset = datasets.CIFAR10(data_dir, train=True,
                                     transform=transforms.Compose(transform))
    test_dataset = datasets.CIFAR10(data_dir, train=False,
                                    transform=transforms.Compose(transform[2:]))

    self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=test_batch_size,
                                                   shuffle=False)

    if first_epoch_batch_size != batch_size:
      self.first_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=first_epoch_batch_size,
                                                      shuffle=True)
    else:
      self.first_loader = self.train_loader
