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

import time

import numpy as np

from cont_speech_experiment import ContinuousSpeechExperiment
from correlation_metrics import plot_metrics, register_act
from nupic.research.support import parse_config


class SparseCorrExperiment(object):
    def __init__(self, config_file):
        self.dense_network = "denseCNN2"
        self.sparse_network = "sparseCNN2"
        self.config_init = parse_config(config_file)

    def model_comparison(self, plot_results=True, freeze_linear=False, shuffled=False):
        odcorrs, dcorrs = [], []
        oddotproducts, ddotproducts = [], []
        output = [odcorrs, dcorrs, oddotproducts, ddotproducts]

        if shuffled:
            shodcorrs, shdcorrs = [], []
            shoddotproducts, shddotproducts = [], []
            sh_output = [shodcorrs, shdcorrs, shoddotproducts, shddotproducts]

        for exp in [self.dense_network, self.sparse_network]:
            config = self.config_init[exp]
            config["name"] = exp

            if freeze_linear:
                config["freeze_params"] = "output"
            else:
                config["freeze_params"] = []

            outputs = self.run_experiment(config, shuffled=shuffled)
            if shuffled:
                [output[k].append(outputs[k]) for k in range(len(outputs[0]))]
                [sh_output[k].append(sh_output[k]) for k in range(len(outputs[1]))]
            else:
                [output[k].append(outputs[k]) for k in range(len(outputs))]

        plot_metrics(outputs)
        if shuffled:
            plot_metrics(sh_outputs)
            return output, sh_output
        else:
            return output

    def act_fn_comparison(self, plot_results=True, freeze_linear=False, shuffled=False):
        cnn_weight_sparsities = [(1., 1.), (0.5, 0.2)]
        linear_weight_sparsities = [(1.,), (0.1,)]
        cnn_percent_on = [(0.095, 0.125), (1., 1.)]
        linear_percent_on = [(0.1,), (1.,)]
        exp = self.sparse_network

        odcorrs, dcorrs = [], []
        oddotproducts, ddotproducts = [], []
        output = [odcorrs, dcorrs, oddotproducts, ddotproducts]

        if shuffled:
            shodcorrs, shdcorrs = [], []
            shoddotproducts, shddotproducts = [], []
            sh_output = [shodcorrs, shdcorrs, shoddotproducts, shddotproducts]

        for i in range(2):
            for j in range(2):
                config = self.config_init[exp]
                config["name"] = exp

                config["cnn_weight_sparsity"] = cnn_weight_sparsities[i]
                config["weight_sparsity"] = linear_weight_sparsities[i]
                config["cnn_percent_on"] = cnn_percent_on[j]
                config["linear_percent_on"] = linear_percent_on[j]

                if freeze_linear:
                    config["freeze_params"] = "output"
                else:
                    config["freeze_params"] = []

                outputs = self.run_experiment(config, shuffled=shuffled)
                if shuffled:
                    [output[k].append(outputs[k]) for k in range(len(outputs[0]))]
                    [sh_output[k].append(sh_output[k]) for k in range(len(outputs[1]))]
                else:
                    [output[k].append(outputs[k]) for k in range(len(outputs))]

        leg = ["dense + k-winner", "dense + ReLU", "sparse + k-winner", "sparse + ReLU"]
        plot_metrics(outputs, legend_=leg)
        if shuffled:
            plot_metrics(sh_output, legend_=leg)
            return output, sh_output
        else:
            return output
            
    def layer_size_comparison(self, layer_sizes, plot_results=True,
                              freeze_linear=False, shuffled=False):
        # get a factor to multiply the weight sparsity and percent on with
        sparse_factor = [layer_sizes[0] / k for k in layer_sizes]

        # get the default sparsity in the config file to multiply with "sparse_factor"
        curr_sparsity = self.config_init["sparseCNN2"]["cnn_weight_sparsity"]
        curr_percent_on = self.config_init["sparseCNN2"]["cnn_percent_on"]

        odcorrs, dcorrs = [], []
        oddotproducts, ddotproducts = [], []
        output = [odcorrs, dcorrs, oddotproducts, ddotproducts]

        if shuffled:
            shodcorrs, shdcorrs = [], []
            shoddotproducts, shddotproducts = [], []
            sh_output = [shodcorrs, shdcorrs, shoddotproducts, shddotproducts]

        for exp in [self.dense_network, self.sparse_network]:
            for ind in range(len(layer_sizes)):
                config = self.config_init[exp]
                config["name"] = exp

                if freeze_linear:
                    config["freeze_params"] = "output"
                else:
                    config["freeze_params"] = []

                config["cnn_out_channels"] = (layer_sizes[ind], layer_sizes[ind])
                config["cnn_weight_sparsity"] = (
                    curr_sparsity[0] * sparse_factor[ind],
                    curr_sparsity[1] * sparse_factor[ind])
                config["cnn_percent_on"] = (
                    curr_percent_on[0] * sparse_factor[ind],
                    curr_percent_on[1] * sparse_factor[ind])

                outputs = self.run_experiment(
                    config, layer_sizes[ind], shuffled=shuffled)
                if shuffled:
                    [output[k].append(outputs[k]) for k in range(len(outputs[0]))]
                    [sh_output[k].append(sh_output[k]) for k in range(len(outputs[1]))]
                else:
                    [output[k].append(outputs[k]) for k in range(len(outputs))]

        leg = list(zip(np.repeat(["denseCNN", "sparseCNN"], len(layer_sizes)),
                   3 * layer_sizes)
        plot_metrics(outputs, legend_=leg)
        if shuffled:
            plot_metrics(sh_output, legend_=leg)
            return output, sh_output
        else:
            return output

    def run_experiment(self, config, layer_size=None, shuffled=False):
        experiment=ContinuousSpeechExperiment(config=config)
        start_time=time.time()
        experiment.train_entire_dataset(0)
        end_time=np.round(time.time() - start_time, 3)
        if layer_size is not None:
            print("{} layer size network trained in {} s".format(layer_size, end_time))

        if shuffled:
            corrs, shuffled_corrs=register_act(experiment)
            return corrs, shuffled_corrs
        else:
            corrs=register_act(experiment)
            return corrs
