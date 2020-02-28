import sys
sys.path.append('../')

from sparse_speech_experiment import *
import configparser
from nupic.research.support import parse_config


config_file = "../experiments.cfg"
exp = "sparseCNN2"
config_init = parse_config(config_file)
config = config_init[exp]
config["name"] = exp
config["seed"] = 42
config["data_dir"] = "/Users/afisher/nta/nupic.torch/examples/gsc/data/"

experiment = SparseSpeechExperiment(config=config)

n_epochs = 10

for epoch in range(n_epochs):
    experiment.train(epoch)
    experiment.validate()
    
experiment.test()