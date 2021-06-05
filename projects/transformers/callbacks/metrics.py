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
from transformers import TrainerCallback


class TrackEvalMetrics(TrainerCallback):
    """
    This callback is used to store a time series of metrics (e.g. accuracy),
    after trainer.evaluate() is called. It is designed to provide the same
    metrics for training and validation sets, at the same time points.
    """
    def __init__(self):
        """
        Set up two dictionaries to track training and eval metrics, and a list
        to track steps.

        *_metrics dictionaries take metric names (e.g. 'loss') as keys
        and have a list that tracks that metric accross time

        Example:
            self.eval_metrics['acc'] -> [acc1, acc2, ..., accn]
            self.steps = [eval_steps, eval_steps*2, ..., eval_steps*n]
        """
        self.eval_metrics = {}
        self.train_metrics = {}  # ignore since this could slow down training
        self.steps = []
        self.step_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Update steps counter, training metrics, & eval metrics"""

        is_mnli_mm = self.check_for_mnli(metrics)
        if not is_mnli_mm:
            self.step_counter += args.eval_steps
            self.steps.append(self.step_counter)

            # update eval_results
            for key in metrics.keys():
                if key not in self.eval_metrics:
                    self.eval_metrics[key] = [metrics[key]]
                else:
                    self.eval_metrics[key].append(metrics[key])

            # Possibly update train_results

            # Possibly wandb logging

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Stop from accumulating results accross multiple runs
        """
        self.__init__()

    def check_for_mnli(self, metrics):

        # Using a workaround for mnli, and only evaluating mnli-mm at the very
        # end of training. Therefore don't update anything
        is_mnli_mm = False
        for key in metrics.keys():
            if key == "mm_accuracy":
                is_mnli_mm = True
                break

        if is_mnli_mm:
            self.mm_metrics = {}
            for key in metrics.keys():
                if "_" in key:
                    raw_key = key.split("_", 1)[1]  # "mm_accuracy" -> "accuracy"
                else:
                    raw_key = key
                self.mm_metrics[raw_key] = metrics[key]

        return is_mnli_mm
