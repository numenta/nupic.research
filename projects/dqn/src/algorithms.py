# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
import torch

from .misc import Scheduler
from .networks import DQNNetwork


class AlgorithmBase():
    pass


class DQN(AlgorithmBase):
    """
    DQN algorithm implementation compatible with Vernon
    Parts of the code adapted from:
    https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py
    """

    def __init__(
        self,
        observation_size,
        num_actions,
        num_steps,
        device,
        q_function_network_class=DQNNetwork,
        exploration_rate_scheduler_class=Scheduler,
        q_function_optimizer=torch.optim.Adam,
        discount_rate=0.99,
        q_function_lr=1e-2,
        exploration_rate=5e-2,
        exploration_rate_final=1e-3,
        target_function_update_rate=1e-2,
        target_function_update_frequency=100,
        clip_error=True,
    ):

        self.device = device

        # Initialize networks
        input_channels = observation_size[2]
        self.q_function = q_function_network_class(
            input_channels=input_channels,
            num_actions=num_actions
        )
        self.q_function.to(self.device)
        self.target_q_function = q_function_network_class(
            input_channels=input_channels,
            num_actions=num_actions
        )
        self.target_q_function.to(self.device)

        # Initialize optimizer
        self.q_function_optimizer = q_function_optimizer(
            self.q_function.parameters(), lr=q_function_lr
        )

        # Initialize e-greedy scheduler
        self.exploration_rate_scheduler = exploration_rate_scheduler_class(
            start_rate=exploration_rate,
            end_rate=exploration_rate_final,
            max_steps=num_steps
        )

        self.num_actions = num_actions
        self.clip_error = clip_error
        self.discount_rate = discount_rate
        self.target_function_update_rate = target_function_update_rate
        self.target_function_update_frequency = target_function_update_frequency

    def gradient_step(self, experiences):
        self.q_function_optimizer.zero_grad()
        output, loss = self.compute_error(experiences)
        output.backward(loss)
        self.q_function_optimizer.zero_grad()
        return loss.sum().item()

    def state_dict(self):
        return {
            "q_function": self.q_function.state_dict(),
            "target_q_function": self.target_q_function.state_dict(),
            "q_function_optimizer": self.q_function_optimizer.state_dict(),
            "exploration_rate_scheduler": self.exploration_rate_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.q_function.load_state_dict(state_dict["q_function"])
        self.target_q_function.load_state_dict(state_dict["target_q_function"])
        self.q_function_optimizer.load_state_dict(state_dict["q_function_optimizer"])
        self.exploration_rate_scheduler.load_state_dict(
            state_dict["exploration_rate_scheduler"])

    def get_next_action(self, last_observation):
        if np.random.random() < self.exploration_rate_scheduler():
            return self.q_function.get_action(last_observation)
        else:
            return np.random.randint(self.num_actions)

    def unpack(self, experiences):
        obs, act, rew, done, next_obs, info = zip(*experiences)
        # Action and rewards are scalars
        act_batch = torch.tensor(act, device=self.device)
        rew_batch = torch.tensor(rew, device=self.device)
        # Observations are tensors, with shape (1, *observation_size)
        obs_batch = torch.cat(obs).to(device=self.device)
        next_obs_batch = torch.cat(next_obs).to(device=self.device)
        # Done are booleans
        not_done_mask = torch.tensor(done, device=self.device) == False  # noqa: E712

        return obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask

    def compute_error(self, experiences):

        # Unpack experiencs
        obs_batch, act_batch, rew_batch, next_obs_batch, not_done_mask = \
            self.unpack(experiences)

        # Compute current Q value, q_func takes only state and output value for every
        # state-action pair. We choose Q based on action taken.
        # self.q_function(obs_batch) 128, 4
        # current_q_values 128
        current_q_values = (self.q_function(obs_batch)
                            .gather(1, act_batch.unsqueeze(1)).squeeze())

        # Compute next Q value based on which action gives max Q values. Detach variable
        # from the current graph since we don't want gradients for next Q to propagated
        # next_max_q and next_q_values is 128
        next_max_q = self.target_q_function(next_obs_batch).detach().max(1)[0]
        next_q_values = not_done_mask * next_max_q

        # Compute the target of the current Q values
        # rew_batch target_q_values 128
        target_q_values = rew_batch + (self.discount_rate * next_q_values)

        # Compute Bellman error
        bellman_error = target_q_values - current_q_values
        if self.clip_error:
            bellman_error = bellman_error.clamp(-1, 1)

        # Note: bellman_delta * -1 will be right gradient for minimization
        loss = bellman_error * -1.0

        return current_q_values, loss

    def post_step(self, num_steps):
        self.update_target_function(num_steps)

    def update_target_function(self, num_steps):
        if num_steps % self.target_function_update_frequency == 0:
            update_rate = self.target_function_update_rate
            for param, target_param in zip(
                self.q_function.parameters(), self.target_q_function.parameters()
            ):
                target_param.data *= (1 - update_rate)
                target_param.data += param * update_rate
