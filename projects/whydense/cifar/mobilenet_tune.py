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
import click
import ray
import ray.tune as tune
import torch
from torchvision import datasets

from nupic.research.frameworks.pytorch.image_transforms import *
from nupic.research.support.parse_config import parse_config
from projects.whydense.cifar.mobilenet_cifar import MobileNetCIFAR10



class MobileNetTune(tune.Trainable):
  """
    ray.tune trainable class for running MobileNet CIFAR experiments:
  """


  def _setup(self, config):
    import ray
    import ray.tune as tune
    import os
    import torch.nn as nn
    from nupic.research.frameworks.pytorch.models import (MobileNetV1,
                                                          MobileNetV1SparsePoint,
                                                          MobileNetV1SparseDepth)

    # Cannot pass functions to "ray.tune". Make sure to use string in the config
    # and evaluate during "_setup"
    config["loss_function"] = eval(config["loss_function"], globals(), locals())
    config["model_type"] = eval(config["model_type"], globals(), locals())
    self.experiment = MobileNetCIFAR10(config)


  def _train(self):
    self.experiment.train(self._iteration)
    return self.experiment.test()


  def _save(self, checkpoint_dir):
    return self.experiment.save(checkpoint_dir)


  def _restore(self, checkpoint_dir):
    self.experiment.restore(checkpoint_dir)



@ray.remote
def run_experiment(config, trainable, num_cpus=1, num_gpus=0):
  """
  Run a single tune experiment in parallel as a "remote" function.

  :param config: The experiment configuration
  :type config: dict
  :param trainable: tune.Trainable class with your experiment
  :type trainable: :class:`ray.tune.Trainable`
  """
  resources_per_trial = {"cpu": num_cpus, "gpu": num_gpus}
  print("experiment =", config["name"])
  print("resources_per_trial =", resources_per_trial)

  # Stop criteria. Default to total number of iterations/epochs
  stop_criteria = {"training_iteration": config.get("iterations")}
  stop_criteria.update(config.get("stop", {}))
  print("stop_criteria =", stop_criteria)

  tune.run(
    trainable,
    name=config["name"],
    stop=stop_criteria,
    config=config,
    resources_per_trial=resources_per_trial,
    num_samples=config.get("repetitions", 1),
    local_dir=config.get("path", None),
    upload_dir=config.get("upload_dir", None),
    sync_function=config.get("sync_function", None),
    checkpoint_freq=config.get("checkpoint_freq", 0),
    checkpoint_at_end=config.get("checkpoint_at_end", False),
    export_formats=config.get("", None),
    search_alg=config.get("search_alg", None),
    scheduler=config.get("scheduler", None),
    verbose=config.get("verbose", 2),
    resume=config.get("resume", False),
    queue_trials=config.get("queue_trials", False),
    reuse_actors=config.get("reuse_actors", False),
    trial_executor=config.get("trial_executor", None),
    raise_on_failed_trial=config.get("raise_on_failed_trial", True)
  )



@click.command()
@click.option("-c", "--config", type=open, default="mobilenet_experiments.cfg",
              show_default=True, help="your experiments config file")
@click.option("-e", "--experiments", multiple=True,
              help="run only selected experiments, by default run all "
                   "experiments in config file.")
@click.option("-n", "--num_cpus", type=int, default=os.cpu_count(),
              show_default=True, help="number of cpus you want to use")
@click.option("-g", "--num_gpus", type=int, default=torch.cuda.device_count(),
              show_default=True, help="number of gpus you want to use")
@click.option("--redis-address", help="Ray Cluster redis address")
def main(config, experiments, num_cpus, num_gpus, redis_address):
  print("config =", config.name)
  print("experiments =", experiments)
  print("num_gpus =", num_gpus)
  print("num_cpus =", num_cpus)
  print("redis_address =", redis_address)

  # Use configuration file location as the project location.
  projectDir = os.path.dirname(config.name)
  projectDir = os.path.abspath(projectDir)
  print("projectDir =", projectDir)

  # Load and parse experiment configurations
  configs = parse_config(config, experiments, globals=globals())

  # Pre-download dataset
  data_dir = os.path.join(projectDir, "data")
  datasets.CIFAR10(data_dir, download=True, train=True)

  # Initialize ray cluster
  if redis_address is not None:
    ray.init(redis_address=redis_address, include_webui=True)
    num_cpus = 1
  else:
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, local_mode=num_cpus == 1)

  # Run all experiments in parallel
  results = []
  for exp in configs:
    config = configs[exp]
    config["name"] = exp

    # Make sure local directories are relative to the project location
    path = config.get("path", None)
    if path and not os.path.isabs(path):
      config["path"] = os.path.join(projectDir, path)

    data_dir = config.get("data_dir", "data")
    if not os.path.isabs(data_dir):
      config["data_dir"] = os.path.join(projectDir, data_dir)

    # When running multiple hyperparameter searches on different experiments,
    # ray.tune will run one experiment at the time. We use "ray.remote" to
    # run each tune experiment in parallel as a "remote" function and wait until
    # all experiments complete
    results.append(run_experiment.remote(config, MobileNetTune,
                                         num_cpus=1,
                                         num_gpus=num_gpus / num_cpus))

  # Wait for all experiments to complete
  ray.get(results)

  ray.shutdown()



if __name__ == "__main__":
  main()
