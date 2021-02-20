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
import pickle

import pandas as pd


def results_to_markdown(model_name, results_files):
    results = {}
    for results_file in results_files:
        with open(results_file, "rb") as f:
            results.update(pickle.load(f))
    print(results)

    report_results = {t: r.to_string() for t, r in results.items()}
    consolidated_results = {t: r.consolidate() for t, r in results.items()}

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
    args = parser.parse_args()
    results_to_markdown(**args.__dict__)
