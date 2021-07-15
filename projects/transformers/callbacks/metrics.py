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
# import wandb
from transformers import TrainerCallback

from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params


class TrackEvalMetrics(TrainerCallback):
    """
    This callback is used to store a time series of metrics (e.g. accuracy),
    after trainer.evaluate() is called. It is designed to provide the same
    metrics for training and validation sets, at the same time points.
    """
    def __init__(self, sparsity_tol=None):
        """
        Set up two dictionaries to track training and eval metrics, and a list
        to track steps.

        *_metrics dictionaries take metric names (e.g. 'loss') as keys
        and have a list that tracks that metric accross time

        Example:
            self.eval_metrics['acc'] -> [acc1, acc2, ..., accn]
            self.steps = [eval_steps, eval_steps*2, ..., eval_steps*n]
        """
        self.sparsity_tolerance = sparsity_tol if sparsity_tol is not None else 0.01
        self.eval_metrics = {}
        self.eval_metrics["sparsity"] = []
        self.eval_metrics["num_total_params"] = []
        self.eval_metrics["num_nonzero_params"] = []
        self.eval_metrics["lr"] = []
        # TODO: track train_metrics, ignore for now
        self.train_metrics = {}
        self.steps = []
        self.step_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Update steps counter, training metrics, & eval metrics"""

        is_mnli_mm = self.check_for_mnli(metrics)
        if not is_mnli_mm:
            self.step_counter += args.eval_steps
            self.steps.append(self.step_counter)

            # track performance metrics
            for key in metrics.keys():
                if key not in self.eval_metrics:
                    self.eval_metrics[key] = [metrics[key]]
                else:
                    self.eval_metrics[key].append(metrics[key])

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
                assert abs(sparse_diff) < self.sparsity_tolerance, "Model sparsity"
                f"fluctuated beyond acceptable range. {self.eval_metrics['sparsity']}"

            # track learning rate
            # get_last_lr() returns lr for each parameter group. For now,
            # assume lrs are the same for all and just track one.
            if kwargs["lr_scheduler"] is not None:
                last_lr = kwargs["lr_scheduler"].get_last_lr()
                self.eval_metrics["lr"].append(last_lr[0])

            # if wandb.run is not None:
            #     print("logging wandb stuff here")
            #     wandb.run.summary.update(self.eval_metrics)  # or [-1]
            # if self.step_counter > 100:
            #     import pdb
            #     pdb.set_trace()
            # # wandb.log(self.eval_metrics, commit=False)
            # TODO
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
        is_mnli_mm = "mm_accuracy" in metrics.keys()

        # flake8 wants this to be a def, not a lambda
        def ksplit(k):
            return k.split("_", 1)[1] if "_" in k else k

        if is_mnli_mm:
            self.mm_metrics = {ksplit(k): v for k, v in metrics.items()}

        return is_mnli_mm

    # TODO
    # Aggregate data on train end or at least make it an option
    # this would make hyperparameter tuning easier.
