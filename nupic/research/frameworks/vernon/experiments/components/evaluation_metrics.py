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


class ContinualLearningMetrics(object):
    """
    Mixin that encapsulates different continual learning metrics.
    Currently covers basic methods.
    To include: forgetting ratio, backward transfer, forward transfer.
    """

    def eval_current_task(self):
        """
        Evaluates accuracy at current task only. Used for debugging.
        """
        return self.validate(self.current_task)

    def eval_first_task(self):
        """
        Evaluates accuracy at first task only. Used for debugging.
        """
        return self.validate(tasks=0)

    def eval_all_visited_tasks(self):
        """
        Evaluates all tasks seen so far jointly. Equivalent to average accuracy
        """
        return self.validate(tasks=range(self.current_task + 1))

    def eval_all_tasks(self):
        """
        Evaluates all tasks, including visited and not visited tasks.
        """
        return self.validate(tasks=range(self.num_tasks))

    def eval_individual_tasks(self):
        """
        Most common scenario in continual learning.
        Evaluates all tasks seen so far, and report accuracy individually.
        """
        task_results = {}
        for task_id in range(self.current_task + 1):
            for k, v in self.validate(tasks=task_id).items():
                task_results[f"task{task_id}__{k}"] = v
        return task_results
