#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
"""
If a finetuning run included multiple runs per task, and you only want to save
the run that had the highest scores, this script will go through and remove
run_* directories that should have been deleted previously, but weren't.
"""

import os
import pickle
import sys

import wandb

from run_utils import rm_prefixed_subdirs


def get_best_run_and_rm_others(exp_name):

    # Load task_results.p
    results_dir = os.path.expanduser("~/nta/results/experiments/transformers")
    results_path = os.path.join(results_dir, exp_name)
    results_file = os.path.join(results_path, "task_results.p")
    with open(results_file, "rb") as f:
        task_results = pickle.load(f)

    # remove run_* directories one task at a time
    for task, results in task_results.items():
        task_dir = os.path.join(results_path, task)
        best_run = results.get_model_with_best_max()
        skip = f"run_{best_run}"
        print(f"Removing run_ directories in {task_dir}")
        rm_prefixed_subdirs(task_dir, prefix="run_", skip=skip)


if __name__ == "__main__":

    # set up logging
    project = os.getenv("WANDB_PROJECT", "huggingface")
    run_name = "cleanup"
    wandb.init(project=project, name=run_name)

    # each directory is one experiment (possible with multiple tasks)
    directories = list(sys.argv[1:])
    failed_directories = []

    # try cleaning up experiments one at a time
    for d in directories:
        try:
            get_best_run_and_rm_others(d)
        except BaseException:
            # log that you were unable to rm runs for this one
            print(f"The following directory could not be cleaned: {d}")
            failed_directories.append(d)

    # log which directories were not properly cleaned up
    if len(failed_directories) == 0:
        print("All directories were properly cleaned")
    else:
        print(f"Done cleaning up. The following directories could not be "
              "cleaned: ")
        for fd in failed_directories:
            print(fd)
