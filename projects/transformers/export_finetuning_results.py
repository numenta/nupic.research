# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
"""
Parse results folder and prints it as markdown to add it to leaderboard
Requires as arguments path to one or more task_results.p files and
an optional model name (-m)

Can parse several results file at once
It is gonna parse in order, so later entries in the list will update earlier entries.
Useful if you only need to rerun one or more tasks instead of all
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TaskResultsAnalysis:

    def _get_time_and_label(self, run_idx):

        if "steps" in self.all_results[run_idx].keys():
            x = self.all_results[run_idx]['steps']
            xlabel = "steps"
            if self.training_args is not None:
                suffix = f"\n(batch_size={self.training_args.per_device_train_batch_size})"
                xlabel = xlabel + suffix
        elif "epoch" in self.all_results[run_idx].keys():
            x = self.all_results[run_idx]['epoch']
            xlabel = "epoch"
        else:
            print("Warning, unknown time metric")
            key0 = list(self.all_results[run_idx].keys())
            x = np.arange(len(self.all_results[run_idx][key0]))
            xlabel = ""

        return x, xlabel


    def plot_run(self, run_idx, metric, save_name=False):
        """Plot one metric on one run over time"""

        fig, ax = plt.subplots()
        y = self.all_results[run_idx][metric]
        x, xlabel = self._get_time_and_label(run_idx)
        ax.plot(x, y, ".", ms=10, linestyle='dashed')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} on run {run_idx}")

        if save_name:
            plt.savefig(save_name)

        print("Figure ready, hit plt.show() to visualize")

        return fig, ax


    def plot_metric(self, metric, save_name=False):
        """Plot one metric across all runs, you type plt.show()"""

        fig, ax = plt.subplots()
        for run_idx in range(len(self.all_results)):
            x, xlabel = self._get_time_and_label(run_idx)
            y = self.all_results[run_idx][metric]

            ax.plot(x, y, ".", linestyle='dashed', label=f"{run_idx}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} on all runs")
        plt.legend()

        if save_name:
            plt.savefig(save_name)

        print("Figure ready, hit plt.show() to visualize")

        return fig, ax



def results_to_markdown(results_files, model_name, reduction):
    results = {}
    for results_file in results_files:
        if os.path.isdir(results_file):
            results_file = os.path.join(results_file, "task_results.p")
        with open(results_file, "rb") as f:
            results.update(pickle.load(f))

    # Aggregate using chosen reduction method
    for _, task_results in results.items():
        task_results.reduce_metrics(reduction=reduction)

    # Get results in string format and consolidated per task
    report_results = {t: r.to_string() for t, r in results.items()}
    consolidated_results = {t: r.consolidate() for t, r in results.items()}

    # Calculate totals for bert and glue
    num_tasks_bert = len(results) - 1 if "wnli" in results else len(results)
    average_bert = sum(
        [value for task, value in consolidated_results.items() if task != "wnli"]
    ) / num_tasks_bert
    average_glue = sum(consolidated_results.values()) / len(results)

    report_results["average_bert"] = f"{average_bert*100:.2f}"
    report_results["average_glue"] = f"{average_glue*100:.2f}"

    df = pd.DataFrame.from_dict({model_name: report_results}).transpose()
    print(df.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", type=str, nargs="+",
                        help="Path to pickle file with finetuning results")
    parser.add_argument("-m", "--model_name", type=str,
                        default="model",
                        help="Name of the model to save")
    parser.add_argument("-r", "--reduction", type=str,
                        default="max", choices=["mean", "max"],
                        help="Reduction method to use to aggregate results"
                             "from multiple runs")
    args = parser.parse_args()
    results_to_markdown(**args.__dict__)
