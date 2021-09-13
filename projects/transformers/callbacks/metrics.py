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

    def __init__(self, n_eval_sets=1, sparsity_tolerance=0.01):
        """
        Set up a dictionary that tracks evaluation metrics

        *_metrics dictionaries take metric names (e.g. 'loss') as keys
        and have a list that tracks that metric accross time

        Example:
            self.eval_metrics['acc'] -> [acc_1, acc_2, ..., acc_n]
            self.steps = [eval_steps, eval_steps*2, ..., eval_steps*n]

        :param n_eval_sets: int, how many evaluation sets you are evaluating
                            on. Default is one, but could be 2 for mnli.
        :param sparsity_tolerance: float, threshold for absolute value change
                                   in sparsity. If sparsity changes by more
                                   than tolerance, issue a warning.
        """

        self.sparsity_tolerance = sparsity_tolerance
        self.n_eval_sets = n_eval_sets
        self.eval_metrics = {}
        self.eval_metrics["sparsity"] = []
        self.eval_metrics["num_total_params"] = []
        self.eval_metrics["num_nonzero_params"] = []
        self.eval_metrics["lr"] = []
        self.steps = []
        self.step_counter = 0  # how many training steps
        self.call_counter = 0  # how many times on_evaluate is called

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Update eval metrics and possibly step counter, sparsity, and lr"""

        # track performance metrics
        for key in metrics.keys():
            if key not in self.eval_metrics:
                self.eval_metrics[key] = [metrics[key]]
            else:
                self.eval_metrics[key].append(metrics[key])

        self.call_counter += 1

        # Update steps, sparsity, learning rate
        if self.maybe_update():
            self.update_auxillary_metrics(args, **kwargs)

    def maybe_update(self):
        """
        If just one eval set
            update step counter, sparsity, etc.
        Else
            update only if you finished cycling through all eval sets
        """
        do_remaining_updates = False
        if self.n_eval_sets > 1:
            if self.call_counter % self.n_eval_sets == 0:
                do_remaining_updates = True
        else:
            do_remaining_updates = True

        return do_remaining_updates

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Stop from accumulating results accross multiple runs
        """
        self.__init__(self.n_eval_sets, self.sparsity_tolerance)

    def update_auxillary_metrics(self, args, **kwargs):
        """
        Track sparsity, learning rate, and run checks to ensure sparsity
        is not changing too much. Run only after updating metrics for all
        eval sets.
        """
        # track sparsity information
        num_total, num_nonzero = count_nonzero_params(kwargs["model"])
        model_sparsity = 1 - (num_nonzero / num_total)
        self.eval_metrics["num_total_params"].append(num_total)
        self.eval_metrics["num_nonzero_params"].append(num_nonzero)
        self.eval_metrics["sparsity"].append(model_sparsity)

        # guarantee that everything stayed sparse, up to specified tolerance
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

        self.step_counter += args.eval_steps
        self.steps.append(self.step_counter)
