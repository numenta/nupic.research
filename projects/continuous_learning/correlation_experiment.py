#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import torch
import numpy as np

from nupic.research.support import parse_config
from correlation_metrics import register_act, plot_metrics
from cont_speech_experiment import ContinuousSpeechExperiment, ClasswiseDataset

class SparseCorrExperiment(object):
    def __init__(self, config):
        self.dense_network = "denseCNN2"
        self.sparse_network = "sparseCNN2"
        self.default_cnn_size = 64
        self.config_init = parse_config(config_file)
        
    def layer_size_comparison(self, layer_sizes, plot_results=True, freeze_linear=False, shuffled=False):
        sparse_factor = [layer_sizes[0]/k for k in layer_sizes]
        curr_sparsity = config_init["sparseCNN2"]["cnn_weight_sparsity"]
        curr_percent_on = config_init["sparseCNN2"]["cnn_percent_on"]
        
        odcorrs, dcorrs = [], []
        oddotproducts, ddotproducts = [], []
        
        if shuffled:
            shodcorrs, shdcorrs = [], []
            shoddotproducts, shddotproducts = [], []
        
        for exp in [self.dense_network, self.sparse_network]:
            for size in layer_sizes:
                config = self.config_init[exp]
                
                if freeze_linear:
                    config["freeze_params"] = "output"
                else:
                    config["freeze_params"] = []
                    
                config["cnn_out_channels"] = (size,size)
                config["cnn_weight_sparsity"] = (curr_sparsity[0]*sparse_factor, curr_sparsity[1]*sparse_factor)
                config["cnn_percent_on"] = (curr_percent_on[0]*sparse_factor, curr_percent_on[1]*sparse_factor)
    
                outputs = run_experiment(config, shuffle=shuffled)
                if shuffled:
                    results.append(outputs[0])
                    shuffled_results.append(outputs[1])
                    
        
    def run_experiment(config, shuffled=False):
        experiment = ContinuousSpeechExperiment(config=config)
        start_time = time.time()
        experiment.train_entire_dataset(0)
        end_time = np.round(time.time() - start_time, 3)
        print("{} layer size network trained in {} s".format(size, end_time))

        if shuffled:
            corrs, shuffled_corrs = register_act(experiment)
            return corrs, shuffled_corrs
        else:
            corrs = register_act(experiment)                    
            return corrs                
        