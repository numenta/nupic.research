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

__all__ = [
    "EvalPerTask",
]


class EvalPerTask:
    """
    Mixin for getting metrics, namely accuracy, for each task independently
    """

    def validate(self, loader=None):
        """
        Run validation on the currently active tasks, and gather accuracy measures
        for each of those tasks independently.
        """

        # Figure this out
        if loader is None:
            loader = self.val_loader

        results = {}
        for task in range(self.current_task + 1):
            self.val_loader.sampler.set_active_tasks(task)

            # Store results for each individual task in results[task]
            task_results = super().validate(loader)
            results[task] = task_results

        # Loop over results per task to get averages
        task_average_results = average_over_tasks(results)
        results["task_average"] = task_average_results

        # Flatten dictionary and add task id to key name (e.g. mean_accuracy_4)
        results = format_results(results)

        self.val_loader.sampler.set_active_tasks(self.current_task)

        return results


def format_results(results):
    """
    Flatten results dictionary and add task id to key name

    :param results: Nested dictionary similar to
                    results[task_i] = {metric_name: metric_value}

    Returns dictionary similar to results[metric_name_i] = metric_value
    """
    new_results = {}
    for key in results.keys():
        if key == "task_average":
            for sub_key in results[key]:
                new_results[sub_key] = results[key][sub_key]
        else:
            for sub_key in results[key]:
                new_sub_key = "_".join([sub_key, str(key)])
                new_results[new_sub_key] = results[key][sub_key]

    return new_results


def average_over_tasks(results):
    """
    Sum metrics for each task and divide by n_tasks to get average results.
    Assumes number of samples per task is the same for all, as is the case for
    permutedMNIST.
    """
    # Get all metrics
    keys = list(results.keys())
    metrics = list(results[keys[0]].keys())

    # loop over tasks and average
    average_metrics = {}
    for metric in metrics:
        total = 0
        for key in keys:
            total += results[key][metric]

        mean = total / len(keys)
        average_metrics[metric] = mean

    return average_metrics
