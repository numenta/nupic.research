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

import ray
from ray import tune
from ray.tune.schedulers import *
from torchvision import datasets

from nupic.research.frameworks.pytorch.model_utils import *
from nupic.research.frameworks.pytorch.image_transforms import *
from nupic.research.frameworks.pytorch.cifar_experiment import TinyCIFAR

import configparser

import logging

# Remove annoying messages saying training is taking too long
logging.getLogger("ray.tune.util").setLevel(logging.ERROR)


def trial_name_string(trial):
  """
  Args:
      trial (Trial): A generated trial object.

  Returns:
      trial_name (str): String representation of Trial.
  """
  s = str(trial)
  chars = "{}[]() ,="
  for c in chars:
    s = s.replace(c, "_")

  if len(s) > 85:
    s = s[0:75] + "_" + s[-10:]
  return s


class CIFARTune(TinyCIFAR, tune.Trainable):
  """
  ray.tune trainable class for running small CIFAR models:
  - Override _setup to reset the experiment for each trial.
  - Override _train to train and evaluate each epoch
  - Override _save and _restore to serialize the model
  """

  def __init__(self, config=None, logger_creator=None):
    TinyCIFAR.__init__(self)
    tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)


  def _setup(self, config):
    """Custom initialization.

    Args:
        config (dict): Hyperparameters and other configs given.
            Copy of `self.config`.
    """
    self.model_setup(config)


  def _train(self):
    """Implement train() for a single epoch.

    Returns:
        A dict that describes training progress."""

    ret = self.train_epoch(self._iteration)
    print("epoch", self._iteration, ":", ret)
    return ret


  def _save(self, checkpoint_dir):
    return self.model_save(checkpoint_dir)


  def _restore(self, checkpoint):
    """Subclasses should override this to implement restore().

    Args:
        checkpoint (str | dict): Value as returned by `_save`.
            If a string, then it is the checkpoint path.
    """

    self.model_restore(checkpoint)


  def _stop(self):
    """Subclasses should override this for any cleanup on stop."""
    if self._iteration < self.iterations:
      print("CIFARTune: stopping early at epoch {}".format(self._iteration))


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

  print("gpu usage:",config.get("num_gpus", 0) / config.get("num_cpus", 1))

  tune.run(
    trainable,
    name=config["name"],
    local_dir=config["path"],
    stop=stop_criteria,
    config=config,
    num_samples=config.get("repetitions", 1),
    search_alg=config.get("search_alg", None),
    scheduler=config.get("scheduler",
                         MedianStoppingRule(
                           time_attr="training_iteration",
                           reward_attr='noise_accuracy',
                           min_samples_required=3,
                           grace_period=20,
                           verbose=False,
                         )),
    trial_name_creator=tune.function(trial_name_string),
    trial_executor=config.get("trial_executor", None),
    checkpoint_at_end=config.get("checkpoint_at_end", False),
    checkpoint_freq=config.get("checkpoint_freq", 0),
    resume=config.get("resume", False),
    reuse_actors=config.get("reuse_actors", False),
    verbose=config.get("verbose", 0),
    resources_per_trial={
      "cpu": 1, "gpu": config.get("num_gpus", 0) / config.get("num_cpus", 1)
    }
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
                         default=os.cpu_count()-1,
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

  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  train_dataset = datasets.CIFAR10(data_dir, download=True, train=True)

  # Initialize ray cluster
  ray.init(num_cpus=options.num_cpus,
           num_gpus=options.num_gpus,
           local_mode=options.num_cpus == 1)

  # Run all experiments in parallel
  results = []
  for exp in configs:
    config = configs[exp]
    config["name"] = exp
    config["num_cpus"] = options.num_cpus
    config["num_gpus"] = options.num_gpus

    # Make sure local directories are relative to the project location
    path = config.get("path", "results")
    if not os.path.isabs(path):
      config["path"] = os.path.join(projectDir, path)

    data_dir = config.get("data_dir", "data")
    if not os.path.isabs(data_dir):
      config["data_dir"] = os.path.join(projectDir, data_dir)

    # When running multiple hyperparameter searches on different experiments,
    # ray.tune will run one experiment at the time. We use "ray.remote" to
    # run each tune experiment in parallel as a "remote" function and wait until
    # all experiments complete
    results.append(run_experiment.remote(config, CIFARTune))

  # Wait for all experiments to complete
  ray.get(results)

  ray.shutdown()
