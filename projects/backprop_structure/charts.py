# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import json
import os
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ray import tune

from nupic.research.frameworks.backprop_structure.networks import (
    gsc_lesparsenet,
    mnist_lesparsenet,
)
from nupic.research.frameworks.backprop_structure.ray_ax import (
    get_best_config,
    get_frontier_trials,
)

NOISE_LEVELS = np.arange(0.0, 0.51, 0.05)


def get_noise_score(results, custom_noise_levels=None):
    scores = []

    for result in results:
        logdir = result["logdir"]
        with open(Path(logdir) / "result.json", "r") as fin:
            final_result = json.loads(fin.readlines()[-1])
        if "noise_results" in final_result:
            if custom_noise_levels is not None:
                scores.append(
                    sum(final_result["noise_results"][str(noise_level)]["total_correct"]
                        for noise_level in custom_noise_levels))
            else:
                scores.append(sum(final_result["noise_score"]))

    return np.mean(scores)


def get_hsd_weights_by_layer(hsd_constructor):
    lesparsenet = hsd_constructor()
    hsd_weights_by_layer = {
        "output": lesparsenet.output.weight.detach().numpy().size,
    }
    for name in ["cnn1_cnn", "cnn2_cnn", "linear1"]:
        m = getattr(lesparsenet, name)
        hsd_weights_by_layer[name] = (m.weight_sparsity
                                      * m.module.weight.detach().numpy().size)
    return hsd_weights_by_layer


def save_charts(  # NOQA: C901
        chart_prefix, experiments, dense_exp, densenet_constructor,
        hsd_exp, hsd_kw_exp, hsd_constructor, error_xlim, error_ylim,
        acc_xlim, acc_ylim, acc_noise_xlim):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = Path(script_dir) / "output"
    os.makedirs(output_dir, exist_ok=True)

    plot1_results = []
    plot2_results = []

    noise_plot1_results = []
    noise_plot2_results = []

    for exp_name in experiments:
        m = import_module(f"runs.{exp_name}")

        frontier_trials = get_frontier_trials(
            os.path.expanduser(f"~/ray_results/{exp_name}"),
            m.PARAMETERS,
            m.NUM_TRAINING_ITERATIONS)

        nz = np.array([np.mean([result["inference_nz"] for result in results])
                       for config, results in frontier_trials])
        acc = np.array([np.mean([result["mean_accuracy"] for result in results])
                        for config, results in frontier_trials])
        err = 1 - acc
        noise_score = np.array([get_noise_score(results, NOISE_LEVELS)
                                for config, results in frontier_trials])

        order = np.argsort(nz)
        plot1_results.append((nz[order], err[order]))
        plot2_results.append((nz[order], acc[order]))
        noise_plot1_results.append((nz[order], noise_score[order]))

        order = np.argsort(acc)
        noise_plot2_results.append((acc[order], noise_score[order]))

    if dense_exp is not None:
        m = import_module(f"runs.{dense_exp}")
        _, results = get_best_config(os.path.expanduser(f"~/ray_results/{dense_exp}"),
                                     m.PARAMETERS,
                                     m.NUM_TRAINING_ITERATIONS)
        densenet_accuracy = np.mean([result["mean_accuracy"] for result in results])
        densenet_noisescore = get_noise_score(results, NOISE_LEVELS)

        densenet = densenet_constructor(
            cnn_activity_percent_on=(1.0, 1.0),
            cnn_weight_percent_on=(1.0, 1.0),
            linear_activity_percent_on=(1.0,),
            linear_weight_percent_on=(1.0,),)
        densenet_num_weights = sum(
            getattr(densenet, name).weight.detach().numpy().size
            for name in ["cnn1_cnn", "cnn2_cnn", "linear1_linear", "output"])

    if hsd_exp is not None:
        m = import_module(f"runs.{hsd_exp}")
        df = tune.Analysis(
            os.path.expanduser(f"~/ray_results/{hsd_exp}")).dataframe()
        df = df[df["training_iteration"] == m.NUM_TRAINING_ITERATIONS]
        hsd_accuracy = np.mean(df["mean_accuracy"])
        hsd_noisescore = np.mean(get_noise_score(
            [result for _, result in df.iterrows()],
            NOISE_LEVELS))

    if hsd_kw_exp is not None:
        m = import_module(f"runs.{hsd_kw_exp}")
        df = tune.Analysis(
            os.path.expanduser(f"~/ray_results/{hsd_kw_exp}")).dataframe()
        df = df[df["training_iteration"] == m.NUM_TRAINING_ITERATIONS]
        hsd_kw_accuracy = np.mean(df["mean_accuracy"])
        hsd_kw_noisescore = np.mean(get_noise_score(
            [result for _, result in df.iterrows()],
            NOISE_LEVELS))

        hsd_weights_by_layer = get_hsd_weights_by_layer(hsd_constructor)
        hsd_num_weights = sum(hsd_weights_by_layer.values())

    fig = plt.figure(figsize=(4, 4))

    for nz, err in plot1_results:
        plt.plot(nz, err, "-o")

    if hsd_exp is not None:
        plt.plot(hsd_num_weights, 1 - hsd_accuracy, "x", color="C1")
    if hsd_kw_exp is not None:
        plt.plot(hsd_num_weights, 1 - hsd_kw_accuracy, "x", color="C1")
    if dense_exp is not None:
        plt.plot(densenet_num_weights, 1 - densenet_accuracy, "d", color="C3")

    plt.xlabel("# of weights")
    plt.xscale("log")
    plt.xlim(error_xlim)

    plt.ylabel("error rate")
    plt.yscale("log")
    plt.ylim(error_ylim)
    plt.grid(True)
    plt.tight_layout()
    filename = output_dir / f"{chart_prefix}_error_rate.pdf"
    print(f"Saving {filename}")
    fig.savefig(filename)

    fig = plt.figure(figsize=(4, 4))
    for nz, acc in plot2_results:
        plt.plot(nz, acc, "-o")

    if hsd_exp is not None:
        plt.plot(hsd_num_weights, hsd_accuracy, "x", color="C1")
    if hsd_kw_exp is not None:
        plt.plot(hsd_num_weights, hsd_kw_accuracy, "x", color="C1")
    if dense_exp is not None:
        plt.plot(densenet_num_weights, densenet_accuracy, "d", color="C3")

    plt.xlabel("# of weights")
    plt.xscale("log")
    plt.xlim(acc_xlim)
    plt.ylabel("accuracy")
    plt.ylim(acc_ylim)
    plt.grid(True)
    plt.tight_layout()
    filename = output_dir / f"{chart_prefix}_accuracy.pdf"
    print(f"Saving {filename}")
    fig.savefig(filename)

    fig = plt.figure(figsize=(4, 4))
    for nz, noise_score in noise_plot1_results:
        plt.plot(nz, noise_score, "-o")

    if hsd_exp is not None:
        plt.plot(hsd_num_weights, hsd_noisescore, "x", color="C1")
    if hsd_kw_exp is not None:
        plt.plot(hsd_num_weights, hsd_kw_noisescore, "x", color="C1")
    if dense_exp is not None:
        plt.plot(densenet_num_weights, densenet_noisescore, "d", color="C3")

    plt.xlabel("# of weights")
    plt.xscale("log")
    plt.xlim(acc_xlim)

    plt.ylabel("noise score")

    plt.grid(True)
    plt.tight_layout()
    filename = output_dir / f"{chart_prefix}_noise.pdf"
    print(f"Saving {filename}")
    fig.savefig(filename)

    fig = plt.figure(figsize=(4, 4))
    for acc, noise_score in noise_plot2_results:
        plt.plot(acc, noise_score, "-o")

    if hsd_exp is not None:
        plt.plot(hsd_accuracy, hsd_noisescore, "x", color="C1")
    if hsd_kw_exp is not None:
        plt.plot(hsd_accuracy, hsd_kw_noisescore, "x", color="C1")
    if dense_exp is not None:
        plt.plot(densenet_accuracy, densenet_noisescore, "d", color="C3")

    plt.xlabel("accuracy")
    plt.xlim(acc_noise_xlim)

    plt.ylabel("noise score")

    plt.grid(True)
    plt.tight_layout()
    filename = output_dir / f"{chart_prefix}_acc_noise.pdf"
    print(f"Saving {filename}")
    fig.savefig(filename)


if __name__ == "__main__":
    save_charts(
        chart_prefix="mnist",
        experiments=["ax_ln_bps_mnist"],
        dense_exp="ax_ln_mnist",
        densenet_constructor=mnist_lesparsenet,
        hsd_exp="run_lenet_staticstructure_mnist",
        hsd_kw_exp="run_lenet_staticstructure_kwinners_mnist",
        hsd_constructor=mnist_lesparsenet,
        error_xlim=(1e1, 2e6),
        error_ylim=(1e-3, 1e0),
        acc_xlim=(1e3, 2e6),
        acc_ylim=(0.9875, 1.0),
        acc_noise_xlim=(0.98, 1.0),
    )

    save_charts(
        chart_prefix="gsc",
        experiments=["ax_ln_bps_gsc"],
        dense_exp="ax_ln_gsc",
        densenet_constructor=gsc_lesparsenet,
        hsd_exp="run_lenet_staticstructure_gsc",
        hsd_kw_exp="run_lenet_staticstructure_kwinners_gsc",
        hsd_constructor=gsc_lesparsenet,
        error_xlim=(1e1, 2e6),
        error_ylim=(1e-2, 1e0),
        acc_xlim=(1e3, 2e6),
        acc_ylim=(0.955, 0.98),
        acc_noise_xlim=(0.95, 0.98),
    )

    save_charts(
        chart_prefix="mnist_batchnorm",
        experiments=["ax_ln_bps_mnist", "ax_ln_bps_batchnorm_mnist"],
        dense_exp="ax_ln_mnist",
        densenet_constructor=mnist_lesparsenet,
        hsd_exp=None,
        hsd_kw_exp=None,
        hsd_constructor=mnist_lesparsenet,
        error_xlim=(1e1, 2e6),
        error_ylim=(1e-3, 1e0),
        acc_xlim=(1e3, 2e6),
        acc_ylim=(0.9875, 1.0),
        acc_noise_xlim=(0.98, 1.0),
    )
