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
from scipy import stats

from finetuning_constants import REPORTING_METRICS_PER_TASK


class TaskResultsAnalysis:

    def __init__(self, task_results_dict, run_name=None):
        """
        run_utils.TaskResults contains data for a single task, but spans multiple
        runs. TaskResultsAnalysis takes a list of TaskResults objects, so you can
        analyze multiple tasks and multiple runs per task.

        e.g.
            self.task_results_dict["wnli"] returns a TaskResults object

        Note that __getitem__ is overridden so you can also just type
            self["wnli"] to get a TaskResults object
        """

        self.task_results_dict = task_results_dict
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = None

        # Do not run an analysis without checking sparsity first
        for task in task_results_dict.keys():
            self.verify_sparsity(task)

    def __getitem__(self, key):
        return self.task_results_dict[key]

    def _get_time_and_label(self, task, run_idx):

        if "steps" in self[task].all_results[run_idx].keys():
            x = self[task].all_results[run_idx]["steps"]
            xlabel = "steps"
            if self[task].training_args is not None:
                suffix = "\n(batch_size="
                f"{self[task].training_args.per_device_train_batch_size})"
                xlabel = xlabel + suffix
        elif "epoch" in self[task].all_results[run_idx].keys():
            x = self[task].all_results[run_idx]["epoch"]
            xlabel = "epoch"
        else:
            print("Warning, unknown time metric")
            key0 = list(self[task].all_results[run_idx].keys())
            x = np.arange(len(self[task].all_results[run_idx][key0]))
            xlabel = ""

        return x, xlabel

    def plot_run(self, task, run_idx, metric, save_name=False, fig=None, ax=None):
        """Plot one metric on one run over time"""

        if not ax:
            fig, ax = plt.subplots()

        y = self[task].all_results[run_idx][metric]
        x, xlabel = self._get_time_and_label(task, run_idx)
        ax.plot(x, y, ".", ms=10, linestyle="dashed", alpha=.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.set_title(f"{task}: {metric} on run {run_idx}")

        if save_name:
            plt.savefig(save_name)

        print("Figure ready, hit plt.show() to visualize")

        return fig, ax

    def plot_metric(self, task, metric, save_name=False, fig=None, ax=None):
        """Plot one metric across all runs, you type plt.show()"""

        if not ax:
            fig, ax = plt.subplots()

        if task not in self.task_results_dict:
            print(f"skipping {task} in run {self.run_name}")
            return fig, ax

        for run_idx in range(len(self[task].all_results)):
            x, xlabel = self._get_time_and_label(task, run_idx)
            y = self[task].all_results[run_idx][metric]

            ax.plot(x, y, ".", linestyle="dashed", label=f"run: {run_idx}", alpha=0.7)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.set_title(f"{task}: {metric} on all runs")
        plt.legend()

        if save_name:
            plt.savefig(save_name)

        print("Figure ready, hit plt.show() to visualize")

        return fig, ax

    """
    No quality assurance or testing below this line within this class. These
    methods are being rapidly prototyped and will be revised greedily as
    needed for further analysis.
    """

    def trajectory_stats(self, task, run_idx, metric, start=None, stop=None):

        if start is None:
            start = 0
        if stop is None:
            trajectory = np.array(
                self[task].all_results[run_idx][metric][start:])
        else:
            trajectory = np.array(
                self[task].all_results[run_idx][metric][start:stop])

        descriptive_stats = stats.describe(trajectory)
        print(descriptive_stats)

        return descriptive_stats

    def gather_trajectory_stats(self, task, start=None, stop=None):

        self[task].trajectory_stats = []
        for run_idx in range(len(self[task].all_results)):
            self[task].trajectory_stats.append({})
            for metric in self[task].all_results[run_idx].keys():
                descriptive_stats = self.trajectory_stats(
                    task, run_idx, metric, start, stop)
                self[task].trajectory_stats[-1][metric] = descriptive_stats

    def get_stats_by_timepoint(self, task, metric, idx):

        time_point_across_runs = []
        for run_idx in range(len(self[task].all_results)):
            point = self[task].all_results[run_idx][metric][idx]
            time_point_across_runs.append(point)

        descriptive_stats = stats.describe(time_point_across_runs)

        return np.array(time_point_across_runs), descriptive_stats

    def get_all_stats_by_timepoint(self, task, metric):

        # TODO
        pass

    def get_metrics_from_runs(self, task):
        """
        Create a 2d numpy array where each row represents a run. The
        row contains a trajectory for a given metric.

        e.g.
            metric_all_runs[2] could return eval_loss at every time step
            on the 3rd run for a given task.
        """
        metrics = [m for m in self[task].reporting_metrics]

        if hasattr(self[task], "score_matrix"):
            return self[task].score_matrix

        n_steps = len(self[task].all_results[0]["steps"])
        n_runs = len(self[task].all_results)

        metrics_all_runs = []
        for _ in range(len(self[task].reporting_metrics)):
            metrics_all_runs.append(np.zeros((n_runs, n_steps)))

        for m in range(len(metrics)):
            for run_idx in range(n_runs):
                metrics_all_runs[m][run_idx, :] = np.array(
                    self[task].all_results[run_idx][metrics[m]])

        self[task].score_matrix = metrics_all_runs

        return metrics_all_runs

    def get_best_scores_per_run(self, task):
        metrics = self[task].reporting_metrics
        metrics_all_runs = self.get_metrics_from_runs(task)
        if len(metrics) > 1:
            metric_all_runs = np.zeros_like(metrics_all_runs[0])
            for m in range(len(metrics)):
                metric_all_runs += metrics_all_runs[m]
            metric_all_runs /= len(metrics)
        else:
            metric_all_runs = metrics_all_runs[0]
        best_idx_per_run = np.argmax(metric_all_runs, axis=1)
        best_avg_metric_scores = metric_all_runs[np.arange(len(self[task])),
                                                 best_idx_per_run]

        best_metric_scores = []
        for m in range(len(metrics)):
            data = metrics_all_runs[m][np.arange(len(self[task])), best_idx_per_run]
            best_metric_scores.append(data)
        return best_avg_metric_scores, best_metric_scores

    def get_best_scores_best_run(self, task):
        best_avg_metric_scores, best_scores = self.get_best_scores_per_run(task)
        best_idx = np.argmax(best_avg_metric_scores)
        best_score = best_avg_metric_scores[best_idx]
        best_metric_scores = [best_score[best_idx] for best_score in best_scores]
        return best_score, best_metric_scores

    def max_scores(self):

        maxes = {}
        for task in self.task_results_dict.keys():
            maxes[task] = self.get_best_scores_best_run(task)

        _max_scores = {task: maxes[task][0] for task in maxes.keys()}
        str_scores = dict()  # f"{self.results[m]*100:.2f}"
        for task in maxes.keys():
            metrics = maxes[task][1]
            str_list_score = [f"{metrics[i]*100:.2f}" for i in range(len(metrics))]
            str_score = "/".join(str_list_score)
            str_scores[task] = str_score
            print(str_score)

        # Calculate totals for bert and glue
        num_tasks_bert = len(maxes) - 1 if "wnli" in maxes else len(maxes)
        average_bert = sum(
            [value for task, value in _max_scores.items() if task != "wnli"]
        ) / num_tasks_bert
        average_glue = sum(_max_scores.values()) / len(_max_scores)

        return average_bert, average_glue, str_scores

    def verify_sparsity(self, task):

        if "sparsity" in self[task].all_results[0].keys():
            self.plot_metric(task, "sparsity")
        else:
            print(f"sparsity was not tracked for this model: {self.run_name}")

    def glue_load_best_model_at_end(self, reduction="max"):
        """
        Compute the glue score as though load_best_model_at_end were true.
        Just rely on existing code in TaskResults to do this by flipping
        class attributes.
        """
        pass


def compare_models(dict_of_task_analyses, tasks, metric, save_prefix=None):
    """
    Compare a series of models using a single metric. Each model can
    include data about multiple tasks and multiple runs.

    Arguments
        dict_of_task_analyses: dict  --  key points to model, value
            points to a TaskResultsAnalysis object

        tasks: list of str  --  e.g. ['wnli', 'mrpc'], must key into
            TaskResultsAnalysis objects

        metric: str  --  name of measure to compare with, e.g. "eval_loss"
    """

    if isinstance(tasks, str):
        tasks = [tasks]

    n_models = len(dict_of_task_analyses)
    xwidth = 5 * n_models
    for task in tasks:

        if n_models > 5:
            n_rows = int(np.ceil(n_models / 4))
            n_cols = 4
            xwidth = 5 * 4
            yheight = n_rows * 10
            fig, ax = plt.subplots(n_rows,
                                   n_cols,
                                   figsize=(xwidth, yheight),
                                   sharex=True,
                                   sharey=True)
            c = 0
            row = 0
            col = 0
            for model in dict_of_task_analyses.keys():
                row = c // 4
                col = c % 4
                _, _ = dict_of_task_analyses[model].plot_metric(
                    task,
                    metric,
                    ax=ax[row, col]
                )

                plt.legend()

                ttl = ax[row, col].get_title()
                ttl = ttl + f"\n{model}"
                ax[row, col].set_title(ttl)
                c = c + 1

        else:
            fig, ax = plt.subplots(1,
                                   n_models,
                                   figsize=(xwidth, 10),
                                   sharex=True,
                                   sharey=True)

            c = 0
            for model in dict_of_task_analyses.keys():
                _, _ = dict_of_task_analyses[model].plot_metric(
                    task,
                    metric,
                    ax=ax[c]
                )

                plt.legend()

                ttl = ax[c].get_title()
                ttl = ttl + f"\n{model}"
                ax[c].set_title(ttl)
                c = c + 1

        if save_prefix:
            save_name = os.path.join(save_prefix, f"{task}_{metric}_simple_no_esc.png")
            plt.tight_layout()
            plt.savefig(save_name)


def merge_data_to_results(results_file, results=None):

    if not results:
        results = {}

    print_message_1 = "Length of results for {0} prior to update: " \
                      "{1} in file {2}"

    print_message_2 = "Length of results for {0} after update: " \
                      "{1} in file {2}"

    with open(results_file, "rb") as f:
        data = pickle.load(f)

    if results == {}:
        results.update(data)
        for task in data.keys():
            print(print_message_1.format(
                task,
                len(results[task].all_results),
                os.path.split(results_file)[-2])
            )
        return results

    for task in data.keys():
        if task in results:
            print(print_message_1.format(
                task,
                len(results[task].all_results),
                os.path.split(results_file)[-2]))
            for run in data[task].all_results:
                results[task].all_results.append(run)
            print(print_message_2.format(
                task,
                len(results[task].all_results),
                os.path.split(results_file)[-2]))
        else:
            results[task] = data[task]

    return results


def adapt_old_mnli(results):
    """
    TaskResults class has been changing over time. Need to normalize the
    attributes for consitent results / comparison.
    """
    for task in results.keys():
        task_results = results[task]
        task_results.reporting_metrics = REPORTING_METRICS_PER_TASK[task]
        for run in range(len(task_results.all_results)):
            if "mm_eval_accuracy" in task_results.all_results[run]:
                task_results.all_results[run]["eval_mm_accuracy"] = \
                    task_results.all_results[run]["mm_eval_accuracy"]

    results["mnli"].best_metric_key = "eval_mm_accuracy"

    return results


def load_results(results_files):
    results = {}
    for results_file in results_files:
        if os.path.isdir(results_file):
            results_file = os.path.join(results_file, "task_results.p")
        results = merge_data_to_results(results_file, results)

    results = adapt_old_mnli(results)

    return results


def load_milestone_info(pretrained_model):
    """
    Look for the milestones table so you can include the eval loss
    during pretraining for the pretrained model that the current experiment
    derives from. If you can't find the pretrained model in the table, just
    leave it as NaN.
    """
    # If pretrained_model wasn't specified as a command line arg
    if not pretrained_model:
        return np.nan

    # Load the milestones dataframe
    milestones_path = os.path.abspath("../results/milestone1.csv")
    milestones_df = pd.read_csv(milestones_path)

    # Look for eval loss for this model and set to NaN if you can't find it
    row_matches = pretrained_model == milestones_df["Model Name"]
    if sum(row_matches) > 0:
        idx = np.argmax(row_matches)
        eval_loss = milestones_df["Eval Loss"].iloc[idx]
    else:
        print(f"Pretrained model name {pretrained_model} not found in "
              "milestones dataframe. Setting eval_loss to NaN.")
        eval_loss = np.nan

    return eval_loss


def results_to_df(results, reduction, model_name, eval_loss):

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

    # Format dataframe with model_name as a column, as this doesn't
    # affect markown printing, but it makes csv io easier.
    df = pd.DataFrame.from_dict({model_name: report_results}).transpose()
    df["model_name"] = df.index
    df = df.reset_index(drop=True)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]  # Reorder so model_name is first column
    df = df[cols]

    # Plug in eval loss for pretrained model or set to NaN if N/A
    df["MLM Eval Loss"] = eval_loss

    return df


def process_results(results_files, model_name, pretrained_model,
                    reduction, csv, md):

    results = load_results(results_files)

    # load a csv file if specified
    csv_df = None
    if len(csv) > 0:
        if os.path.exists(csv):
            print(f"...loading data from {csv}")
            csv_df = pd.read_csv(csv)

    eval_loss = load_milestone_info(pretrained_model)

    # Aggregate using chosen reduction method
    df = results_to_df(results, reduction, model_name, eval_loss)

    # print markdown
    print(df.to_markdown(index=False))

    # create a new csv file to store results
    if csv_df is None:
        print(f"saving results to a new file: {csv}")
        df.to_csv(os.path.abspath(csv), index=False)
    # merge csv file with current results
    else:
        df = pd.concat([csv_df, df], ignore_index=True)
        df.to_csv(csv, index=False)

    # save a markdown file
    if len(md) > 0:
        df.to_markdown(os.path.abspath(md), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_files", type=str, nargs="+",
                        help="Path to pickle file with finetuning results")
    parser.add_argument("-m", "--model_name", type=str,
                        default="model",
                        help="Name of the model to save")
    parser.add_argument("-p", "--pretrained_model", type=str,
                        default="None", help="Name of pretrained model to "
                        "look for in milestones table")
    parser.add_argument("-r", "--reduction", type=str,
                        default="max", choices=["mean", "max"],
                        help="Reduction method to use to aggregate results"
                             "from multiple runs")
    parser.add_argument("-csv", "--csv", type=str,
                        default="",
                        help="Path to a csv file you want results to go to."
                             "If it exists, it will update the csv,"
                             "and if not, it will create a new file")
    parser.add_argument("-md", "--md", type=str, default="",
                        help="Path to a markdown file. Will overwrite.")
    args = parser.parse_args()
    process_results(**args.__dict__)
