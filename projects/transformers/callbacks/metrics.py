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
import logging

from transformers import TrainerCallback

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params


class TrackEvalMetrics(TrainerCallback):
    """
    This callback is used to store a time series of metrics (e.g. accuracy),
    after trainer.evaluate() is called. It is designed to provide the same
    metrics for training and validation sets, at the same time points.
    """
    def __init__(self, eval_sets=None, eval_prefixes=None, sparsity_tolerance=0.01):
        """
        Set up two dictionaries to track training and eval metrics, and a list
        to track steps.

        *_metrics dictionaries take metric names (e.g. 'loss') as keys
        and have a list that tracks that metric accross time

        Example:
            self.eval_metrics['acc'] -> [acc1, acc2, ..., accn]
            self.steps = [eval_steps, eval_steps*2, ..., eval_steps*n]

        This callback also keeps track of model sparsity, and breaks if
        sparsity changes in absolute value by more than sparsity_tolerance.
        If you are using a training approach that sparsifies the model, be sure
        to set sparsity_tolerance to something like 1, so large changes in
        sparsity are accepted.
        """

        self.sparsity_tolerance = sparsity_tolerance
        self.eval_sets = None if eval_sets is None else eval_sets
        self.eval_metrics = {}
        self.eval_metrics["sparsity"] = []
        self.eval_metrics["num_total_params"] = []
        self.eval_metrics["num_nonzero_params"] = []
        self.eval_metrics["lr"] = []
        self.train_metrics = {}
        self.steps = []
        self.step_counter = 0  # how many training steps
        self.call_counter = 0  # how many times on_evaluate is called

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Update steps counter, training metrics, & eval metrics"""

        # track performance metrics
        for key in metrics.keys():
            if key not in self.eval_metrics:
                self.eval_metrics[key] = [metrics[key]]
            else:
                self.eval_metrics[key].append(metrics[key])

        self.call_counter += 1
        do_remaining_updates = False
        if self.eval_sets:
            if len(self.eval_sets) % self.call_counter == 0:
                do_remaining_updates = True
        else:
            do_remaining_updates = True

        if do_remaining_updates:
            self.update_auxillary_metrics(kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Stop from accumulating results accross multiple runs
        """
        self.__init__()

    def update_auxillary_metrics(self, **kwargs):

        # track sparsity information
        num_total, num_nonzero = count_nonzero_params(kwargs["model"])
        model_sparsity = 1 - (num_nonzero / num_total)
        self.eval_metrics["num_total_params"].append(num_total)
        self.eval_metrics["num_nonzero_params"].append(num_nonzero)
        self.eval_metrics["sparsity"].append(model_sparsity)

        # guarantee that everything stayed sparse,
        # up to specified tolerance
        if (self.sparsity_tolerance < 1) and len(self.eval_metrics["sparsity"]) > 1:
            sparse_diff = self.eval_metrics["sparsity"][0] - self.eval_metrics["sparsity"][-1]  # noqa
            if abs(sparse_diff) > self.sparsity_tolerance:
                logging.warn(
                    "Model sparsity fluctuated beyond acceptable range."
                    f"Current sparsity level: {self.eval_metrics['sparsity'][-1]}"
                )

        # track learning rate
        # get_last_lr() returns lr for each parameter group. For now,
        # assume lrs are the same for all and just track one.
        if kwargs["lr_scheduler"] is not None:
            last_lr = kwargs["lr_scheduler"].get_last_lr()
            self.eval_metrics["lr"].append(last_lr[0])

        self.step_counter += 1