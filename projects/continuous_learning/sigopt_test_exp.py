%load_ext autoreload
%autoreload 2

import numpy as np


import torch
import os

from cont_speech_experiment import ContinuousSpeechExperiment, ClasswiseDataset
from nupic.research.support import parse_config
from nupic.research.frameworks.continuous_learning.correlation_metrics import register_act, plot_metrics
from nupic.research.frameworks.continuous_learning.utils import get_act, dc_grad
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptExperiment

from sigopt import Connection

from sigopt_config import sigopt_config as CONFIG

conn = Connection()
experiment = conn.experiments().create(**CONFIG)
print("Created experiment: https://app.sigopt.com/experiment/" + str(experiment.id))

sig_freeze_params = {"True": True, "False": False}


def clear_labels(labels):
    indices = np.arange(11)
    out = np.delete(indices, labels)
    return out

config_file = "experiments.cfg"
with open(config_file) as cf:
    config_init = parse_config(cf)

exp = "sparseCNN2"

layers_ = ["cnn2_kwinner", "linear1_kwinners"]
layer_type = "kwinner"

fcs = []
aucs = []

train_inds = np.arange(1,11).reshape(5,2)
for i in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    params = suggestion.assignments
    
    config = config_init[exp]
    config["name"] = exp
    config["seed"] = np.random.randint(0,200)
    config["cnn_out_channels"] = (params["cnn1_size"], params["cnn2_size"])
    config["cnn_percent_on"] = (params["cnn1_percent_on"], params["cnn2_percent_on"])
    config["cnn_weight_sparsity"] = (params["cnn1_wt_sparsity"], params["cnn2_wt_sparsity"])
                                

    if layer_type == "kwinner":
        config["linear_n"] = (params["linear1_n"],11)
        config["linear_percent_on"] = (params["linear1_percent_on"], params["linear2_percent_on"])
        config["weight_sparsity"] = (params["linear1_weight_sparsity"], params["linear2_weight_sparsity"])
    #     layers_.extend(["linear2_kwinners"])


    config["batch_size"] = 64    
    config["boost_strength"] = 0.0
    config["boost_strength_factor"] = 0.0
    config["duty_cycle_period"] = params["duty_cycle_period"]

    experiment = ContinuousSpeechExperiment(config)

    if layer_type == "dense":
        experiment.model.add_module("output", torch.nn.Linear(1000, 11).cuda())

    freeze_params=layers_

    for j in range(len(train_inds)):

        if j == 0:
            experiment.train(1,train_inds[j],
                            freeze_output=sig_freeze_params[params["freeze_output"]],
                            layer_type=layer_type,
                            output_indices=clear_labels(train_inds[j]))
        else:
            experiment.train(2,train_inds[j],freeze_params=freeze_params,
                            freeze_fun=dc_grad,
                            freeze_pct=params["freeze_pct"],
                            freeze_output=sig_freeze_params[params["freeze_output"]],
                            layer_type=layer_type,
                            output_indices=clear_labels(train_inds[j]))

    fc = experiment.get_forgetting_curve()
    fcs.append(fc)
    auc = experiment.get_auc()
    aucs.append(auc)

    opt_metric = auc
    
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=opt_metric
    )