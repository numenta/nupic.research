import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from cont_speech_experiment import ContinuousSpeechExperiment
from nupic.research.support import parse_config

sys.path.append('../')


data_dir = "/home/ec2-user/nta/data/data_classes/"
config_file = "experiments.cfg"


exp = "sparseCNN2"
config_init = parse_config(config_file)
config = config_init[exp]
config["name"] = exp
config["seed"] = 42
config["data_dir"] = data_dir

experiment = ContinuousSpeechExperiment(config=config)

class_losses = []
ent = []

for label in range(1,11):
    print("training on class {}".format(label))
    for epoch in range(5):
        print("training: epoch {}".format(epoch+1))
        experiment.train(epoch, label)
        
    t = experiment.test()
    class_losses.append(t["mean_accuracy"])
    ent.append(np.round(t["entropy"]))


print(class_losses)
print(ent)
