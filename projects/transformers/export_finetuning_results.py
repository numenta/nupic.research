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

"""

import argparse
import pickle
from collections import defaultdict

import pandas as pd

metrics = {
    "cola": ["eval_matthews_correlation"],
    "mnli": ["eval_accuracy"],
    "mrpc": ["eval_f1", "eval_accuracy"],
    "qnli": ["eval_accuracy"],
    "qqp": ["eval_accuracy", "eval_f1"],
    "rte": ["eval_accuracy"],
    "sst2": ["eval_accuracy"],
    "stsb": ["eval_pearson", "eval_spearmanr"],
    "wnli": ["eval_accuracy"]
}


def results_to_markdown(model_name, results_file):

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    report_results = defaultdict(list)
    consolidated_results = defaultdict(int)
    for task, result_dict in results.items():
        for metric in metrics[task]:
            report_results[task].append(f"{result_dict[metric]*100:.2f}")
            consolidated_results[task] += result_dict[metric]

        report_results[task] = "/".join(report_results[task])
        consolidated_results[task] /= len(metrics[task])

    average_bert = sum(
        [value for task, value in consolidated_results.items() if task != "wnli"]
    ) / 8
    average_glue = sum(consolidated_results.values()) / 9

    report_results["average_bert"] = f"{average_bert*100:.2f}"
    report_results["average_glue"] = f"{average_glue*100:.2f}"
    report_results["mnli"] = f"{report_results['mnli']}/-"

    df = pd.DataFrame.from_dict({model_name: report_results}).transpose()
    print(df.to_markdown())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", type=str,
                        help="Path to pickle file with finetuning results")
    parser.add_argument("model_name", type=str,
                        help="Name of the model to save")
    args = parser.parse_args()
    results_to_markdown(**args.__dict__)
