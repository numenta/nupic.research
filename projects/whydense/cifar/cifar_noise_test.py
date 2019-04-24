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


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def evaluateNoiseSensitivity(model, noise_datasets, layer):
  print("Layer=", layer)
  model.eval()

  with torch.no_grad():
    images = []
    targets = []
    outputs = []
    activations = []

    for d in noise_datasets:

      data, target = d[0]
      data = data.unsqueeze(0)
      images.append(data)
      targets.append(target)

      layer.register_forward_hook(get_activation(layer))

      output = model(data)
      outputs.append(output)
      x = activation[layer]
      x = x.view(-1)
      activations.append(x)

      pred = output.max(1, keepdim=True)[1]
      # correct = pred.eq(target.view_as(pred)).sum().item()

    print(activations[0].shape)
    print(activations[0].nonzero().size(0), activations[1].nonzero().size(0))
    print(activations[0].sum(), activations[1].sum())
    print((activations[0] - activations[1]).abs().sum())
    print(activations[0].dot(activations[0]), activations[0].dot(activations[1]))
    print(images[0].sum(), images[1].sum())
    print(targets[0], targets[1])


def testModel(config, projectDir, checkpoint_path=None):
  """
  Test a pretrained network, specified in this config against noisy vectors
  """
  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  train_dataset = datasets.CIFAR10(data_dir, download=True, train=True)

  # Make sure local directories are relative to the project location
  path = config.get("path", "results")
  if not os.path.isabs(path):
    config["path"] = os.path.join(projectDir, path)

  data_dir = config.get("data_dir", "data")
  if not os.path.isabs(data_dir):
    config["data_dir"] = os.path.join(projectDir, data_dir)

  # Load the pretrained model
  tiny_cifar = TinyCIFAR()
  tiny_cifar.model_setup(config)
  checkpoint_path = checkpoint_path or os.path.join(path, tiny_cifar.model_filename)
  print("Loading model")
  tiny_cifar.model_restore(checkpoint_path)
  print("Done")

  # Create new loaders with batch size 1
  loaders = tiny_cifar._createTestLoaders(tiny_cifar.noise_values, batch_size=1)
  noise_datasets = [l.dataset for l in loaders]

  model = tiny_cifar.model
  modules_to_check = ['cnn_0', 'kwinners_2d_0']
  for m in modules_to_check:
    if m in model._modules.keys():
      evaluateNoiseSensitivity(model=tiny_cifar.model, noise_datasets=noise_datasets,
                               layer=model.__getattr__(m))



def parse_options():
  """ parses the command line options for different settings. """
  optparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  optparser.add_argument("-c", "--config", dest="config", type=open,
                         default="tiny_experiments.cfg",
                         help="your experiments config file")
  optparser.add_argument("-m", "--model_file", dest="checkpoint_path",
                         default=None,
                         help="The checkpoint file for the model")
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


  for exp in configs:
    config = configs[exp]
    config["name"] = exp

    testModel(config,
              projectDir=projectDir,
              checkpoint_path=options.checkpoint_path)
