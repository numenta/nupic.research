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

import torch
import torch.nn.functional as F
from torch import autograd


class ElasticWeightConsolidation:
    """
    Online version of Elastic Weight Consolidation.
    Diagonals of the Fisher information matrix is calculated at the end
    of each task, and used for loss calculation as the subsequent task.

    Based on the implementation in:
    https://github.com/kuc2477/pytorch-ewc

    TODO: Matches original implementation, but requires checking for correctness
    and comparing to existing literature.

    Original implementation only applied to domain task (permuted MNIST), not
    to the task domain.
    """
    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - ewc_lambda: Importance parameter. Multiplier that defines
            the size of the penalty applied to the loss function
            - ewc_fisher_sample_size: Number of samples to be used to
            calculate the fisher matrix.
        """
        super().setup_experiment(config)
        self.ewc_lambda = config.get("ewc_lambda", 40)
        fischer_sampler_size = config.get("ewc_fisher_sample_size",
                                          int(len(self.train_loader) * 0.1))
        self.ewc_fisher_num_batches = fischer_sampler_size // self.batch_size

    def run_task(self):
        """Run outer loop over tasks"""
        ret = super().run_task()
        self.logger.info("Estimating diagonals of the fisher information matrix...")
        self.estimate_fisher()
        return ret

    def estimate_fisher(self):
        """
        Estimate diagonal of the fisher of fisher information matrix.
        Saves mean and diagonals as attributes to the parameter

        TODO: adapt it to take advantage of multiple GPUs/nodes in
        distributed setting.
        """
        loglikelihoods = []
        for idx, (x, y) in enumerate(self.train_loader):
            loglikelihoods.append(
                F.log_softmax(self.model(x))[range(self.batch_size), y.data]
            )
            # Can't use the full dataset, too expensive
            if idx >= self.ewc_fisher_num_batches:
                break

        # Convert into list of tensors per sample
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        # Estimate the fisher information of the parameters.
        grads = zip(*[autograd.grad(
            l, self.model.parameters(),
            retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])

        grads = [torch.stack(gs) for gs in grads]
        fisher_diagonals = [(g ** 2).mean(0) for g in grads]

        for param, fd in zip(self.model.parameters(), fisher_diagonals):
            param.mean_ = param.data.clone()
            param.fisher_ = fd.detach().data.clone()

    def complexity_loss(self, model):
        """
        Defines the complexity loss to be applied.
        """
        cp = super().complexity_loss(model)
        if self.current_task > 0:
            ewc_loss = self.ewc_loss()
        return cp + ewc_loss

    def ewc_loss(self):
        """Note: lambda constant is not divided by 2 in this version"""
        losses = []
        for p in self.model.parameters():
            # Calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((p.fisher_ * (p - p.mean_)**2).sum())

        return self.ewc_lambda * sum(losses)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Initialize EWC attributes")
        eo["complexity_loss"].append("Adds EWC loss if task is greater than 0 ")
        eo["run_task"].append("Estimate diagonals of Fisher matrix at end of task")

        return eo
