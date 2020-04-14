# ----------------------------------------------------------------------
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

import matplotlib.pyplot as plt
import numpy as np

from cont_speech_experiment import ContinuousSpeechExperiment
from nupic.research.frameworks.continuous_learning.correlation_metrics import (
    plot_metrics,
    register_act,
)
from nupic.research.support import parse_config


def train_sequential(experiment, print_acc=False):
    """ Trains a ContinuousSpeechExperiment sequentially,
    i.e. by pairs of labels
    :param experiment: ContinuousSpeechExperiment
    """
    np.random.seed(np.random.randint(0, 100))
    train_labels = np.reshape(np.random.permutation(np.arange(1, 11)), (5, 2))

    epochs = 1  # 1 run per class pair
    entropies, duty_cycles = [], []
    for label in train_labels:
        print("training on class {}".format(label))

        freeze_indices = np.hstack(  # class indices to freeze in output layer
            [0, np.delete(train_labels,
                          np.where(train_labels == label)[0], axis=0).flatten()])

        for epoch in range(epochs):
            experiment.train(epoch, label, indices=freeze_indices)

            mt = experiment.test()
            if print_acc:
                print("Mean accuracy: {}".format(mt["mean_accuracy"]))

            entropies.append(get_entropies(experiment))
            duty_cycles.append(get_duty_cycles(experiment))

    return entropies, duty_cycles


def get_duty_cycles(experiment):
    duty_cycles = []
    for module in experiment.model:
        dc = module.state_dict()
        if "duty_cycle" in dc:
            duty_cycles.append(dc["duty_cycle"].detach().cpu().numpy())

    return duty_cycles


def get_entropies(experiment):
    entropies = []
    for module in experiment.model.modules():
        if module == experiment.model:
            continue
        if hasattr(module, "entropy"):
            entropy = module.entropy()
            entropies.append(entropy.detach().cpu().numpy())

    return entropies


class SparseCorrExperiment(object):
    def __init__(self, config_file):
        self.dense_network = "denseCNN2"
        self.sparse_network = "sparseCNN2"
        self.config_init = parse_config(config_file)
        self.entropies = []
        self.duty_cycles = []

    def reset_ents_dcs(self):
        self.entropies = []
        self.duty_cycles = []

    def model_comparison(self, freeze_linear=False, sequential=False, shuffled=False):
        """ Compare metrics on dense and sparse CNN"""
        self.reset_ents_dcs()
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

            if shuffled:
                outputs, sh_outputs = self.run_experiment(config, sequential=sequential,
                                                          shuffled=True)
                [output[k].append(outputs[k]) for k in range(len(outputs))]
                [sh_output[k].append(sh_outputs[k]) for k in range(len(sh_outputs))]
            else:
                outputs = self.run_experiment(config, sequential=sequential,
                                              shuffled=False)
                [output[k].append(outputs[k]) for k in range(len(outputs))]

        plot_metrics(output)

        if shuffled:
            plot_metrics(sh_output)
            return output, sh_output
        else:
            return output

    def act_fn_comparison(self, freeze_linear=False, sequential=False, shuffled=False):
        """ Compare k-winner and ReLU activations on sparse and dense weights. """
        self.reset_ents_dcs()
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

                if shuffled:
                    outputs, sh_outputs = self.run_experiment(config, shuffled=shuffled)
                    [output[k].append(outputs[k]) for k in range(len(outputs))]
                    [sh_output[k].append(sh_outputs[k]) for k in range(len(sh_outputs))]
                else:
                    outputs = self.run_experiment(config, shuffled=shuffled)
                    [output[k].append(outputs[k]) for k in range(len(outputs))]

        leg = ["dense + k-winner", "dense + ReLU", "sparse + k-winner", "sparse + ReLU"]
        plot_metrics(output, legend_=leg)

        if shuffled:
            plot_metrics(sh_output, legend_=leg)
            return output, sh_output
        else:
            return output

    def layer_size_comparison(self, layer_sizes,
                              compare_models=False,
                              freeze_linear=False,
                              sequential=False,
                              shuffled=False):
        """Get metrics for specified layer sizes
        :param layer_sizes: list of desired CNN layer sizes
        :param compare_models (Boolean): will also run a dense CNN
        """
        self.reset_ents_dcs()
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

        if compare_models:
            experiments = [self.dense_network, self.sparse_network]
        else:
            experiments = [self.sparse_network]

        for exp in experiments:
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

                if shuffled:
                    outputs, sh_outputs = self.run_experiment(
                        config, layer_sizes[ind], shuffled=shuffled)
                    [output[k].append(outputs[k]) for k in range(len(outputs))]
                    [sh_output[k].append(sh_outputs[k]) for k in range(len(sh_outputs))]
                else:
                    outputs = self.run_experiment(
                        config, layer_sizes[ind], shuffled=shuffled)
                    [output[k].append(outputs[k]) for k in range(len(outputs))]

        leg = list(zip(np.repeat(experiments, len(layer_sizes)),
                       3 * layer_sizes))
        plot_metrics(output, legend_=leg)

        if shuffled:
            plot_metrics(sh_output, legend_=leg)
            return output, sh_output
        else:
            return output

    def run_experiment(self, config,
                       layer_size=None,
                       sequential=False,
                       shuffled=False,
                       boosting=True,
                       duty_cycle_period=1000):

        if not boosting:
            config["boost_strength"] = 0.0
            config["boost_strength_factor"] = 0.0

        config["duty_cycle_period"] = duty_cycle_period

        experiment = ContinuousSpeechExperiment(config=config)
        start_time = time.time()
        if sequential:
            entropies, duty_cycles = train_sequential(experiment)
        else:
            experiment.train_entire_dataset(0)

        self.entropies.append(entropies)
        self.duty_cycles.append(duty_cycles)

        end_time = np.round(time.time() - start_time, 3)
        if layer_size is not None:
            print("{} layer size network trained in {} s".format(layer_size, end_time))
        else:
            print("Network trained in {} s".format(end_time))

        if shuffled:
            corrs, shuffled_corrs = register_act(experiment, shuffle=True)
            return corrs, shuffled_corrs
        else:
            corrs = register_act(experiment)
            return corrs


def plot_duty_cycles(experiment):
    for dc in experiment.duty_cycles:
        if len(dc[0]) > 0:
            dc_flat = [item.flatten() for sublist in dc for item in sublist]
            plt.figure(figsize=(10, 12))
            items = ["conv1", "conv2", "linear1"]

            for idx in range(len(items)):
                plt.subplot(2, 2, idx + 1)
                ks = dc_flat[idx::len(items)]
                alpha = 0.12 / np.log(ks[0].shape[0])
                plt.plot(ks, "k.", alpha=0.3)
                plt.plot(ks, "k", alpha=alpha)
                plt.xticks(np.arange(5), np.arange(1, 6))
                plt.ylim((0., 0.2))
                plt.title(items[idx])
                plt.xlabel("Training iteration")
                plt.ylabel("Duty cycle")


def plot_entropies(experiment):
    for ents in experiment.entropies:
        if len(ents[0]) > 0:
            ents_flat = [item.flatten() for sublist in ents for item in sublist]
            plt.figure(figsize=(10, 12))
            items = ["conv1", "conv2", "linear1"]

            for idx in range(len(items)):
                plt.subplot(2, 2, idx + 1)
                ks = ents_flat[idx::len(items)]
                alpha = 0.12 / np.log(ks[0].shape[0])
                plt.plot(ks, "k.", alpha=0.3)
                plt.plot(ks, "k", alpha=alpha)
                plt.xticks(np.arange(5), np.arange(1, 6))
                plt.ylim((0., 0.2))
                plt.title(items[idx])
                plt.xlabel("Training iteration")
                plt.ylabel("Entropy")
