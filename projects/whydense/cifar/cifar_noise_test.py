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


def evaluateSensitivityOneImage(idx, model, noise_datasets, layer,
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



def evaluateNoiseSensitivity(model, noise_datasets, layer_name, numImages=10):
  if layer_name == "image":
    layer = layer_name
  else:
    layer = model.__getattr__(layer_name)

  model.eval()
  with torch.no_grad():

    dots = np.zeros(len(noise_datasets))
    for i in range(numImages):
      evaluateSensitivityOneImage(i, model, noise_datasets, layer, dots)

    print("layer:", layer_name, "mean dots: ", dots / numImages)


def testModel(config, options, projectDir):
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

  evaluateNoiseSensitivity(model=tiny_cifar.model, noise_datasets=noise_datasets,
                           layer_name="image", numImages=options.num_images)

  modules_to_check = ["cnn_0", "avgpool_0", "kwinners_2d_0",
                      "cnn_1", "avgpool_1", "kwinners_2d_1",
                      "flatten", "linear", "kwinners_linear"]
  for m in modules_to_check:
    if m in model._modules.keys():
      evaluateNoiseSensitivity(model=tiny_cifar.model, noise_datasets=noise_datasets,
                               layer_name=m, numImages=options.num_images)



def parse_options():
  """ parses the command line options for different settings. """
  optparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  optparser.add_argument("-c", "--config", dest="config", type=open,
                         default="tiny_experiments.cfg",
                         help="your experiments config file")
  optparser.add_argument("-m", "--model_file", dest="checkpoint_path",
                         default=None,
                         help="The checkpoint file for the model")
  optparser.add_argument("-n", "--num_images", dest="num_images",
                         default=1,
                         help="The number of images to test")
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

    testModel(config, options, projectDir=projectDir)
