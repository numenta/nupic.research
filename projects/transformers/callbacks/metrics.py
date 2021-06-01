import logging

import wandb
from transformers import TrainerCallback

class TrackEvalResults(TrainerCallback):
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
        self.train_metrics = {}
        self.steps = []
        self.step_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Update steps counter, training metrics, & eval metrics"""
        self.step_counter += args.eval_steps
        self.steps.append(self.step_counter)
        print(f"Running TrackEvalResults callback on_evaluate at step {self.steps[-1]}")
        
        # update eval_results
        for key in metrics.keys():
            if key not in self.eval_metrics:
                self.eval_metrics[key] = [metrics[key]]
            else:
                self.eval_metrics[key].append(metrics[key])

        # update train_results

        # wandb logging