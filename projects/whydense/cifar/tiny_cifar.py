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

import argparse
import configparser
import os

import ray
import torch.nn as nn
import torch.optim as optim
from nupic.torch.modules import (KWinners, SparseWeights, Flatten, KWinners2d, rezeroWeights, updateBoostStrength)
from ray import tune
from torchvision import datasets, transforms

from nupic.research.frameworks.pytorch.model_utils import *


def CNNSize(width, kernel_size, padding=1, stride=1):
  return (width - kernel_size + 2 * padding) / stride + 1


class TinyCIFAR(tune.Trainable):
  """
  ray.tune trainable class:
  - Override _setup to reset the experiment for each trial.
  - Override _train to train and evaluate each epoch
  - Override _save and _restore to serialize the model
  """


  def _setup(self, config):

    # Get trial parameters
    seed = config["seed"]
    datadir = config["datadir"]

    # Training parameters
    batch_size = config["batch_size"]
    self.batches_in_epoch = config["batches_in_epoch"]
    first_epoch_batch_size = config["first_epoch_batch_size"]
    self.batches_in_first_epoch = config["batches_in_first_epoch"]

    test_batch_size = config["test_batch_size"]
    self.test_batches_in_epoch = config["test_batches_in_epoch"]

    learning_rate = config["learning_rate"]
    momentum = config["momentum"]

    # Network parameters
    network_type = config["network_type"]
    inChannels, self.h, self.w = config["input_shape"]

    self.boost_strength = config["boost_strength"]
    self.boost_strength_factor = config["boost_strength_factor"]
    self.k_inference_factor = config["k_inference_factor"]

    # CNN parameters - these are lists, one for each CNN layer
    self.cnn_k = config["cnn_k"]
    self.kernel_size = config.get("kernel_size", [3, 3])
    self.out_channels = config.get("out_channels", [32, 32])
    self.in_channels = [inChannels] + self.out_channels

    # Linear parameters
    self.weight_sparsity = config["weight_sparsity"]
    self.linear_n = config["linear_n"]
    self.linear_k = config["linear_k"]
    self.output_size = config.get("output_size", 10)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
      torch.cuda.manual_seed(seed)
    else:
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

    train_dataset = datasets.CIFAR10(datadir, train=True,
                                     transform=self.transform_train)
    test_dataset = datasets.CIFAR10(datadir, train=False,
                                    transform=self.transform_test)

    self.train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True)
    self.test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=test_batch_size, shuffle=True)
    self.first_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=first_epoch_batch_size, shuffle=True)

    if network_type == "tiny_sparse":
      self.createTinySparseModel()


    self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                               momentum=momentum)


  def createTinySparseModel(self):
    prev_w = self.w
    cnn_output_len = []
    for i, (ks,ch) in enumerate(zip(self.kernel_size, self.out_channels)):
      cnn_w = CNNSize(prev_w, ks) // 2
      cnn_output_len.append(int(ch * (cnn_w) ** 2))
      prev_w = cnn_w


    # Create simple sparse model
    self.model = nn.Sequential()

    for c, l in enumerate(cnn_output_len):
      # Sparse CNN layer
      self.model.add_module("cnn_"+str(c),
                            nn.Conv2d(in_channels=self.in_channels[c],
                                      out_channels=self.out_channels[c],
                                      kernel_size=self.kernel_size[c],
                                      padding=1, bias=False))
      self.model.add_module("bn_"+str(c),
                            nn.BatchNorm2d(self.out_channels[c])),
      self.model.add_module("maxpool_"+str(c),
                            nn.MaxPool2d(kernel_size=2))

      if self.cnn_k[c] < 1.0:
        self.model.add_module("kwinners_2d_"+str(c),
                              KWinners2d(n=cnn_output_len[c],
                                         channels=self.out_channels[c],
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


  def _train(self):
    if self._iteration == 0:
      train_loader = self.first_loader
      batches_in_epoch = self.batches_in_first_epoch
    else:
      train_loader = self.train_loader
      batches_in_epoch = self.batches_in_epoch

    trainModel(model=self.model, loader=train_loader,
               optimizer=self.optimizer, device=self.device,
               batches_in_epoch=batches_in_epoch,
               criterion=torch.nn.functional.cross_entropy)
    self.model.apply(rezeroWeights)
    self.model.apply(updateBoostStrength)

    ret = evaluateModel(model=self.model, loader=self.test_loader,
                         batches_in_epoch=self.test_batches_in_epoch,
                         device=self.device,
                         )
    print(self._iteration, ret)
    return ret


  def _save(self, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    torch.save(self.model.state_dict(), checkpoint_path)
    return checkpoint_path


  def _restore(self, checkpoint_path):
    self.model.load_state_dict(checkpoint_path)



@ray.remote
def run_experiment(config, trainable):
  """
  Run a single tune experiment in parallel as a "remote" function.

  :param config: The experiment configuration
  :type config: dict
  :param trainable: tune.Trainable class with your experiment
  :type trainable: :class:`ray.tune.Trainable`
  """
  # Stop criteria. Default to total number of iterations/epochs
  stop_criteria = {
    "training_iteration": config.get("iterations")
  }
  stop_criteria.update(config.get("stop", {}))

  tune.run(
    trainable,
    name=config["name"],
    local_dir=config["path"],
    stop=stop_criteria,
    config=config,
    num_samples=config.get("repetitions", 1),
    search_alg=config.get("search_alg", None),
    scheduler=config.get("scheduler", None),
    trial_executor=config.get("trial_executor", None),
    checkpoint_at_end=config.get("checkpoint_at_end", False),
    checkpoint_freq=config.get("checkpoint_freq", 0),
    resume=config.get("resume", False),
    reuse_actors=config.get("reuse_actors", False),
    verbose=config.get("verbose", 0)
  )



def parse_config(config_file, experiments=None):
  """
  Parse configuration file optionally filtering for specific experiments/sections
  :param config_file: Configuration file
  :param experiments: Optional list of experiments
  :return: Dictionary with the parsed configuration
  """
  cfgparser = configparser.ConfigParser()
  cfgparser.read_file(config_file)

  params = {}
  for exp in cfgparser.sections():
    if not experiments or exp in experiments:
      values = cfgparser.defaults()
      values.update(dict(cfgparser.items(exp)))
      item = {}
      for k, v in values.items():
        try:
          item[k] = eval(v)
        except (NameError, SyntaxError):
          item[k] = v

      params[exp] = item

  return params



def parse_options():
  """ parses the command line options for different settings. """
  optparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  optparser.add_argument("-c", "--config", dest="config", type=open,
                         default="tiny_experiments.cfg",
                         help="your experiments config file")
  optparser.add_argument("-n", "--num_cpus", dest="num_cpus", type=int,
                         default=os.cpu_count(),
                         help="number of cpus you want to use")
  optparser.add_argument("-g", "--num_gpus", dest="num_gpus", type=int,
                         default=torch.cuda.device_count(),
                         help="number of gpus you want to use")
  optparser.add_argument("-e", "--experiment",
                         action="append", dest="experiments",
                         help="run only selected experiments, by default run all experiments in config file.")

  return optparser.parse_args()



if __name__ == "__main__":

  # Load and parse command line option and experiment configurations
  options = parse_options()
  configs = parse_config(options.config, options.experiments)

  # Use configuration file location as the project location.
  # Ray Tune default working directory is "~/ray_results"
  projectDir = os.path.dirname(options.config.name)
  projectDir = os.path.abspath(projectDir)

  # Download dataset once
  datadir = os.path.join(projectDir, "data")
  train_dataset = datasets.CIFAR10(datadir, download=True, train=True)

  # Initialize ray cluster
  ray.init(num_cpus=options.num_cpus,
           num_gpus=options.num_gpus,
           local_mode=options.num_cpus == 1)

  # Run all experiments in parallel
  results = []
  for exp in configs:
    config = configs[exp]
    config["name"] = exp

    # Make sure local directories are relative to the project location
    path = config.get("path", "results")
    if not os.path.isabs(path):
      config["path"] = os.path.join(projectDir, path)

    datadir = config.get("datadir", "data")
    if not os.path.isabs(datadir):
      config["datadir"] = os.path.join(projectDir, datadir)

    # When running multiple hyperparameter searches on different experiments,
    # ray.tune will run one experiment at the time. We use "ray.remote" to
    # run each tune experiment in parallel as a "remote" function and wait until
    # all experiments complete
    results.append(run_experiment.remote(config, TinyCIFAR))

  # Wait for all experiments to complete
  ray.get(results)

  ray.shutdown()
