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

import torch
import torch.nn.functional as F

__all__ = ["SynapticIntelligence"]


class SynapticIntelligence:
    """
    Implementation of Synaptic Intelligence (Zenke, Poole, Ganguli; paper:
    https://arxiv.org/pdf/1703.04200.pdf) that applies the surrogate loss to all feed-
    forward parameters in a neural network. This mixin is only compatible with
    `ContinualLearningExperiment` and its subclasses.

    The config should contain a dict `si_args` that specifies the strength of the SI
    surrogate loss coefficient and damping parameter. By default, SI is not applied to
    dendritic weights, but can be by setting `apply_to_dendrites=True`. If any
    parameters are missing, or if `si_args` is absent, the default values in the
    example below are used.

    Example config:
    ```
    config=dict(
        si_args=dict(
            c=1.0,
            damping=1e-3,
            apply_to_dendrites=False
        )
    )
    ```
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Hyperparameters for surrogate loss
        si_args = config.get("si_args", {})
        self.c = si_args.get("c", 1.0)
        self.damping = si_args.get("damping", 1e-3)
        self.apply_to_dendrites = si_args.get("apply_to_dendrites", False)

        self.small_omega = {}
        self.delta = {}
        self.big_omega = {}

        for name, param in self.named_si_parameters():

            # Track importance weights assigned to each synapse
            self.small_omega[name] = torch.zeros(param.size()).to(self.device)

            # Track how much each parameter moved between tasks
            self.delta[name] = torch.zeros(param.size()).to(self.device)

            # Track regularization strength
            self.big_omega[name] = torch.zeros(param.size()).to(self.device)

        # Unstable parameters are the ones right before the last gradient step (i.e.,
        # they are one gradient step behind the model's current parameters)
        self.unstable_params = {}
        self.update_unstable_params()

        # Stable parameters are the ones converged to after learning each individual
        # task; these are only updated after learning a particular task has completed
        self.stable_params = {}
        self.update_stable_params()

    def run_task(self):
        ret = super().run_task()

        # Compute the following:
        # 1) Delta values - parameter change while learning the most recent task
        # 2) Big omega values - parameter regularization strength in surrogate loss
        for name, param in self.named_si_parameters():

            self.delta[name] = torch.clone(param - self.stable_params[name]).detach()
            self.big_omega[name] += F.relu(self.small_omega[name]) / \
                (self.delta[name] ** 2 + self.damping)

        # Save learned parameters as the new stable parameters
        self.update_stable_params()

        # Reset small omega values
        for name, _param in self.named_si_parameters():
            self.small_omega[name].fill_(0.0)

        return ret

    def error_loss(self, output, target, reduction="mean"):
        """
        Returns the value of the SI loss on the current task: the loss from the
        training objective plus the SI surrogate loss.
        """
        regular_loss = super().error_loss(output, target, reduction)

        # SI regularization term
        surrogate_loss = torch.tensor(0., device=self.device)

        if self.current_task > 1:
            for name, param in self.named_si_parameters():

                big_omega = self.big_omega[name]
                old_param = self.stable_params[name]

                surrogate_loss += (big_omega * ((param - old_param) ** 2)).sum()

        surrogate_loss = self.c * surrogate_loss
        return regular_loss + surrogate_loss

    def post_batch(self, **kwargs):
        """
        Update the running values of small omega variables (i.e., importance weights).
        """
        super().post_batch(**kwargs)

        for name, param in self.named_si_parameters():

            param_change = torch.clone(param - self.unstable_params[name]).detach()
            self.small_omega[name] += -(param.grad * param_change)

        self.update_unstable_params()

    def update_stable_params(self):
        for name, param in self.named_si_parameters():
            self.stable_params[name] = torch.clone(param).detach()

    def update_unstable_params(self):
        for name, param in self.named_si_parameters():
            self.unstable_params[name] = torch.clone(param).detach()

    def named_si_parameters(self):
        """
        Yields the model's `named_parameters` to which SI will be applied.
        """
        for name, param in self.model.named_parameters():
            if self.apply_to_dendrites or "segment" not in name:

                # This line will not be reached if `self.apply_to_dendrites=False` and
                # "segment" occurs in `name`
                yield name, param
