import configparser
import sys

import numpy as np

from nupic.research.support import parse_config
from sparse_speech_experiment import *

sys.path.append('../')


config_file = "../experiments.cfg"
exp = "sparseCNN2"
config_init = parse_config(config_file)
config = config_init[exp]
config["name"] = exp
config["seed"] = 42
# config["data_dir"] = "/Users/afisher/nta/nupic.torch/examples/gsc/data/"
config["data_dir"] = "/Users/afisher/nta/nupic.research/projects/whydense/gsc/notebooks/data_classes/"
# config["mode"]

experiment = SparseSpeechExperiment(config=config)

n_epochs = 0

for epoch in range(n_epochs):
    start_time = time.time()
    print("Epoch: {}".format(epoch))

    experiment.train(epoch)
    end_time = time.time()
    print(experiment.validate())
    print("in {} s".format(np.round(end_time - start_time, 3)))

print("Test: {}".format(experiment.test()))
