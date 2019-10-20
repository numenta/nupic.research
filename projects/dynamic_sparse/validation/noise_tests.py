# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# script to run noise tests

# load the model and the configurations
# could then run a specific model

import json
import os

import numpy as np

import nupic.research.frameworks.dynamic_sparse.models as models
import nupic.research.frameworks.dynamic_sparse.networks as networks
from nupic.research.frameworks.dynamic_sparse.common import browser
from nupic.research.frameworks.dynamic_sparse.common.datasets import Dataset

# load
root = os.path.expanduser("~/nta/results")
experiment_name = "resnet_cifar2"
experiment_path = os.path.join(root, experiment_name)
noise_levels = [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]
results = {}
dataset = None

df = browser.load(experiment_path)

# iterate through all experiment instances
print(experiment_path)
for instance in os.listdir(experiment_path):
    instance_path = os.path.join(experiment_path, instance)
    if os.path.isdir(instance_path):
        print(instance_path)
        # get best epoch
        exp_name = instance[10:][:-28]
        best_epoch = df[df["Experiment Name"] == exp_name]["train_acc_max_epoch"].item()
        # open checkpoint folder
        checkpoint_path = os.path.join(instance_path, "checkpoint_" + str(best_epoch))
        print(checkpoint_path)
        # load config file
        config_path = os.path.join(checkpoint_path, experiment_name + ".json")
        with open(config_path, "r") as file:
            config = json.load(file)
        config["load_from_checkpoint"] = True
        # config['device'] = 'cuda' # only if local
        # load network and model, restore parameters
        network = getattr(networks, config["network"])(config)
        model = getattr(models, config["model"])(network, config)
        model.setup()
        model.restore(checkpoint_path, config["name"])
        # initialize dataset, only once is required
        if dataset is None:
            dataset = Dataset(config)
        # run noise tests
        for noise_level in noise_levels:
            print("noise: ", noise_level)
            accuracies = []
            for _ in range(3):
                dataset.set_noise_loader(noise_level)
                _, acc = model.evaluate_noise(dataset)
                accuracies.append(acc)
            avg_accuracy = np.mean(accuracies)
            print("acc: ", avg_accuracy)
            if exp_name not in results:
                results[exp_name] = {}
            results[exp_name][noise_level] = avg_accuracy

# save results
with open(os.path.join(experiment_path, "noise_results.json"), "w") as file:
    json.dump(results, file)
