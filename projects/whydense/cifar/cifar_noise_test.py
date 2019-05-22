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
import json

from torchvision import datasets

from nupic.research.frameworks.pytorch.model_utils import *
from nupic.research.frameworks.pytorch.image_transforms import *
from nupic.research.frameworks.pytorch.cifar_experiment import (
  TinyCIFAR, create_test_loaders
)
from nupic.research.support.parse_config import parse_config


# This hook records the activations within specific intermediate layers
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def one_image_noise_sensitivity(idx, model, noise_datasets, layer,
                                relative_dot_product_sum):
  """
  For a given image and layer, compute the noise tolerance of this layer for
  each noise dataset. If d_i is the activity of this layer for dataset i, we
  compute the relative dot product for each dataset is computed as:

      relative_dot_product[i] = dot(d_0, d_i) / dot(d_0, d_0)

  The noise_tolerance is added to noise_tolerance_sum:
      relative_dot_product_sum[i] += noise_tolerance[i]

  Here we assume that noise_datasets[0] is a zero noise dataset. If layer ==
  "image" then we compute the noise tolerance of the noisy images themselves. We
  assume the model is already in eval/no_grad state.

  Return a numpy array containing the noise tolerance for each dataset.

  TODO:

    Measure and plot sparsity at each layer. Ideally would like to plot sparsity
    as a function of noise robustness to see if there is any correlation.

    Measure and plot binary overlap at each layer.

    Measure and plot weight sparsity at each layer.
  """
  activations = []
  targets = []
  dot0 = 0.0
  for d, dataset in enumerate(noise_datasets):

    data, target = dataset[idx]
    data = data.unsqueeze(0)
    targets.append(target)

    if layer == "image":
      x = data
    else:
      layer.register_forward_hook(get_activation(layer))
      output = model(data)
      x = activation[layer]

    x = x.view(-1)
    activations.append(x)

    if d == 0:
      dot0 = activations[0].dot(activations[0])

    relative_dot_product_sum[d] += activations[0].dot(x) / dot0

  # print(activations[0].sum(), activations[1].sum())
  # print((activations[0] - activations[1]).abs().sum())
  # print("\nnon-zeros for ",layer,activations[0].nonzero().size(0), activations[1].nonzero().size(0))

  # Ensure the targets are all the same
  assert (len(set(targets)) == 1)

  return relative_dot_product_sum



def evaluateSparsityOneImage(idx, model, noise_datasets, layer,
                             sparsity_sum):
  """
  Given a model and layer, calculate the percent of activations that are
  >0 for each layer for this image for each noise dataset.
  """
  targets = []
  for d, dataset in enumerate(noise_datasets):

    data, target = dataset[idx]
    data = data.unsqueeze(0)
    targets.append(target)

    if layer == "image":
      x = data
    else:
      layer.register_forward_hook(get_activation(layer))
      output = model(data)
      x = activation[layer]

    x = x.view(-1)
    sparsity_sum[d] += (x.nonzero().size(0) / float(x.shape[0]))

  # Ensure the targets are all the same
  assert (len(set(targets)) == 1)

  return sparsity_sum


def one_layer_noise_sensitivity(model, noise_datasets, layer_name, numImages=10):
  if layer_name == "image":
    layer = layer_name
  else:
    layer = model.__getattr__(layer_name)

  model.eval()
  with torch.no_grad():

    dots = np.zeros(len(noise_datasets))
    for i in range(numImages):
      one_image_noise_sensitivity(i, model, noise_datasets, layer, dots)

    print("layer:", layer_name, "mean dots: ", dots / numImages)


def evaluateSparsity(model, noise_datasets, layer_name, numImages=10):
  """
  Given a model and layer, calculate the percent of activations that are
  >0 for each layer across the given number of images for each noise dataset.
  """
  layer = model.__getattr__(layer_name)

  model.eval()
  with torch.no_grad():

    percent_on = np.zeros(len(noise_datasets))
    for i in range(numImages):
      evaluateSparsityOneImage(i, model, noise_datasets, layer, percent_on)

  print("layer:", layer_name, "mean percent_on: ", percent_on / numImages)


def testModelSparsity(config, options, projectDir):
  """
  Test a pretrained network, specified in this config against noisy vectors
  """

  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  _ = datasets.CIFAR10(data_dir, download=True, train=False)

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
  tiny_cifar.model_restore(options.checkpoint_path or
                           os.path.join(path, tiny_cifar.model_filename))

  # Create new loaders with batch size 1
  noise_values = [0.0, 0.1, ]
  loaders = create_test_loaders(noise_values, batch_size=1,
                                data_dir=tiny_cifar.data_dir)
  noise_datasets = [l.dataset for l in loaders]


  # Test noise sensitivity for different layers
  model = tiny_cifar.model

  modules_to_check = ["cnn_0_0", "avgpool_0_0", "kwinners_2d_0_0", "ReLU_0_0",
                      "cnn_1_0", "avgpool_1_0", "kwinners_2d_1_0", "ReLU_1_0",
                      "flatten", "linear_0", "kwinners_linear"]

  for m in modules_to_check:
    if m in model._modules.keys():
      evaluateSparsity(model=model, noise_datasets=noise_datasets,
                       layer_name=m,
                       numImages=options.num_images)


def layer_noise_sensitivities(config, options, projectDir):
  """
  Test a pretrained network, specified in this config against noisy vectors
  """

  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  _ = datasets.CIFAR10(data_dir, download=True, train=False)

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
  tiny_cifar.model_restore(options.checkpoint_path or
                           os.path.join(path, tiny_cifar.model_filename))

  # Create new loaders with batch size 1
  noise_values = [0.0, 0.05, 0.1, 0.15, 0.2]
  loaders = create_test_loaders(noise_values, batch_size=1,
                                data_dir=tiny_cifar.data_dir)
  noise_datasets = [l.dataset for l in loaders]


  # Test noise sensitivity for different layers
  model = tiny_cifar.model

  one_layer_noise_sensitivity(model=tiny_cifar.model, noise_datasets=noise_datasets,
                              layer_name="image", numImages=options.num_images)

  modules_to_check = ["cnn_0_0", "avgpool_0_0", "kwinners_2d_0_0",
                      "cnn_1_0", "avgpool_1_0", "kwinners_2d_1_0",
                      "flatten", "linear_0", "kwinners_linear"]
  for m in modules_to_check:
    if m in model._modules.keys():
      one_layer_noise_sensitivity(model=tiny_cifar.model, noise_datasets=noise_datasets,
                                  layer_name=m, numImages=options.num_images)


def run_noise_tests(config, options, projectDir):
  """
  Test a pretrained network, specified in this config against noisy vectors
  """

  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  _ = datasets.CIFAR10(data_dir, download=True, train=False)

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
  tiny_cifar.model_restore(options.checkpoint_path or
                           os.path.join(path, tiny_cifar.model_filename))

  # Create new loaders with batch size 1
  noise_values = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
  loaders = create_test_loaders(noise_values, batch_size=64,
                                data_dir=tiny_cifar.data_dir)

  print("Running full noise tests using noise values", noise_values)

  ret = tiny_cifar.run_noise_tests(noise_values, loaders, 300)

  print(ret)




def parse_options():
  """ parses the command line options for different settings. """
  optparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  optparser.add_argument("-c", "--config", dest="config", type=str,
                         default="",
                         help="your experiment config file")
  optparser.add_argument("-p", "--params", dest="params", type=str,
                         default="",
                         help="your experiment params json file")
  optparser.add_argument("-m", "--model_file", dest="checkpoint_path",
                         default=None,
                         help="The checkpoint file for the model")
  optparser.add_argument("-n", "--num_images", dest="num_images",
                         default=1, type=int,
                         help="The number of images to test")
  optparser.add_argument("-t", "--noise_tests", dest="noise_tests",
                         default=False, type=bool,
                         help="Run full noise tests on a range of noise values")
  optparser.add_argument("-s", "--sparsity", dest="sparsity",
                         default=False, type=bool,
                         help="Run sparsity tests")
  optparser.add_argument("-l", "--layer_tests", dest="layer_tests",
                         default=False, type=bool,
                         help="Run noise sensitivity analysis on specific layers")
  optparser.add_argument("-e", "--experiment",
                         action="append", dest="experiments",
                         help="run only selected experiments, by default run "
                              "all experiments in config file.")

  return optparser.parse_args()


if __name__ == "__main__":


  print("Torch device count=", torch.cuda.device_count())
  # Load and parse command line option and experiment configurations
  options = parse_options()

  if options.config != "":
    with open(options.config) as f:
      configs = parse_config(f, options.experiments)
    # Use configuration file location as the project location.
    projectDir = os.path.dirname(options.config)

  elif options.params != "":
    with open(options.params) as f:
      params = json.load(f)
      params["data_dir"] = os.path.abspath(os.path.join(".", "data"))
      params["path"] = os.path.abspath(os.path.dirname(options.params))
      configs = {params["name"]: params}
    projectDir = "."

  else:
    raise RuntimeError("Either a .cfg or a params .json file must be specified")

  projectDir = os.path.abspath(projectDir)


  for exp in configs:
    config = configs[exp]
    if "name" not in config: config["name"] = exp

    if options.sparsity:
      testModelSparsity(config, options, projectDir)

    if options.noise_tests:
      run_noise_tests(config, options, projectDir)

    if options.layer_tests:
      layer_noise_sensitivities(config, options, projectDir)
