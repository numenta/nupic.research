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

from torchvision import datasets

from nupic.research.frameworks.pytorch.model_utils import *
from nupic.research.frameworks.pytorch.image_transforms import *
from nupic.research.frameworks.pytorch.cifar_experiment import TinyCIFAR
from nupic.research.support.parse_config import parse_config


def trainModels(configs, projectDir):
  """
  Run all the training experiments specified in configs
  :param configs:
  :param projectDir:
  :return:
  """
  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  train_dataset = datasets.CIFAR10(data_dir, download=True, train=True)


  # Run all experiments in serial
  for exp in configs:
    config = configs[exp]
    config["name"] = exp

    # Make sure local directories are relative to the project location
    path = config.get("path", "results")
    if not os.path.isabs(path):
      config["path"] = os.path.join(projectDir, path)

    data_dir = config.get("data_dir", "data")
    if not os.path.isabs(data_dir):
      config["data_dir"] = os.path.join(projectDir, data_dir)

    model = TinyCIFAR()
    model.model_setup(config)
    for epoch in range(config['iterations']):
      ret = model.train_epoch(epoch)
      print(ret)

    model.model_save(path)


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
                         help="run only selected experiments, by default run "
                              "all experiments in config file.")

  return optparser.parse_args()



if __name__ == "__main__":


  print("Torch device count=", torch.cuda.device_count())
  # Load and parse command line option and experiment configurations
  options = parse_options()
  configs = parse_config(options.config, options.experiments)

  # Use configuration file location as the project location.
  projectDir = os.path.dirname(options.config.name)
  projectDir = os.path.abspath(projectDir)


  trainModels(configs, projectDir=projectDir)