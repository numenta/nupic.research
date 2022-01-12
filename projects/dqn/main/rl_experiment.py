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

import io
import time
from collections import Counter

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler

from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
    serialize_state_dict,
)
from nupic.research.frameworks.vernon.experiments.components.experiment_base import (
    ExperimentBase,
)

from .data_utils import InteractiveDataLoader, ReplayBuffer

__all__ = [
    "OffPolicyReinforcementLearningExperiment",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class InteractiveExperiment(ExperimentBase):
    pass


class OffPolicyReinforcementLearningExperiment(InteractiveExperiment):
    """
    General experiment class used to train off policy RL algorithms
    in interactive datasets.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_steps = 0
        self.current_epoch = 0

    def setup_experiment(self, config):
        super().setup_experiment(config)

        self.device = config.get("device", self.device)

        # Env must be loaded beforehand to get action and observation space
        self.dataset = self.load_dataset(config)

        # Redefine num of actions and observation size based on the dataset
        algorithm_args = config.get("algorithm_args", {})
        try:
            num_actions = self.dataset.action_space.n
        except ValueError:
            self.logger.warning("This implementations expects a discrete action space")
        algorithm_args["num_actions"] = num_actions
        algorithm_args["observation_size"] = \
            self.dataset.observation_space.shape

        # Num steps required for algorithm schedulers
        self.num_steps = config.get("num_steps", None)
        if self.num_steps is None:
            raise ValueError("Must define num_steps")
        algorithm_args["num_steps"] = self.num_steps

        # Define reward clipping, if existing
        self.reward_clip = config.get("reward_clip", None)

        # Initialize algorithm
        algorithm_class = config.get("algorithm_class", None)
        if algorithm_class is None:
            raise ValueError("Must define algorithm_class")

        self.algorithm = algorithm_class(
            **algorithm_args,
            device=self.device
        )

        self.replay_buffer = ReplayBuffer()
        self.buffer_batch_size = config.get("buffer_batch_size", 64)
        self.gradient_steps_per_epoch = config.get("gradient_steps_per_epoch", 16)

        if self.dataset is None:
            raise ValueError("Dataset must be initialized beforehand.")

        self.data_collection_loader = self.create_data_collection_loader()

        self.num_episodes_per_epoch = config.get("num_episodes_per_epoch", 1)

    def collect_data(self, num_episodes):
        """Return the mean of episode rewards"""

        episode_rewards = Counter()
        # how to run one step in data collection?
        for episode in range(num_episodes):
            for experience in self.data_collection_loader:
                observation, action, reward, done, next_observation, info = experience
                # Clip rewards and other transformations from the agent side
                if self.reward_clip is not None:
                    min_reward, max_reward = self.reward_clip
                    reward = max(min_reward, min(reward, max_reward))
                episode_rewards[episode] += reward
                # Append to buffer
                self.replay_buffer.append(experience)

        return np.mean(list(episode_rewards.values()))

    def pre_epoch(self):
        pass

    def post_epoch(self):
        pass

    def run_epoch(self):

        self.pre_epoch()

        # Collect Experiences
        ret = {}
        t0 = time.time()
        ret["average_reward"] = self.collect_data(self.num_episodes_per_epoch)

        # Train
        t1 = time.time()
        ret["average_loss"] = self.train_epoch()
        t2 = time.time()

        self.post_epoch()

        self.logger.debug("data collection time: %s", t1 - t0)
        self.logger.debug("training time: %s", t2 - t1)
        self.logger.debug("---------- End of run epoch ------------")
        self.logger.debug("")

        self.current_epoch += 1
        return ret

    def train_epoch(self):
        # Train loader must be recreated at each step due to different sample subset
        train_loader = self.create_train_loader()
        epoch_loss = 0
        num_samples_in_epoch = 0
        for experiences in train_loader:
            num_samples_in_epoch += len(experiences)
            epoch_loss += self.algorithm.gradient_step(experiences)
            self.post_step()

        self.total_steps += num_samples_in_epoch

        return epoch_loss / num_samples_in_epoch

    def should_stop(self):
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """
        return self.total_steps >= self.num_steps

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        state = {
            "current_epoch": self.current_epoch,
            "total_steps": self.total_steps,
        }

        # Save state into a byte array to avoid ray's GPU serialization issues
        # See https://github.com/ray-project/ray/issues/5519
        with io.BytesIO() as buffer:
            algorithm = self.algorithm
            serialize_state_dict(buffer, algorithm.state_dict())
            state["algorithm"] = buffer.getvalue()

        return state

    def set_state(self, state):
        """
        Restore the experiment from the state returned by `get_state`
        :param state: dictionary with "model", "optimizer", "lr_scheduler", and "amp"
                      states
        """
        if "algorithm" in state:
            with io.BytesIO(state["model"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            self.algorithm.load_state_dict(state_dict)
        if "current_epoch" in state:
            self.current_epoch = state["current_epoch"]
            self.total_steps = state["total_steps"]

    def post_step(self):
        self.algorithm.post_step(self.total_steps)

    def create_train_loader(self):
        """Dataloader used to train the algorithm by sampling from replay buffer"""
        # Reset train loader to only select a subset resampled at each epoch
        subset_indices = np.random.choice(
            np.arange(0, len(self.replay_buffer)),
            size=self.buffer_batch_size * self.gradient_steps_per_epoch
        )

        return DataLoader(
            dataset=self.replay_buffer,
            batch_size=self.buffer_batch_size,
            sampler=SubsetRandomSampler(subset_indices),
            # Return a list of Experiences, not broken down
            collate_fn=lambda x: x
        )

    def create_data_collection_loader(self):
        """Dataloader used to collect data by sampling from dataset"""
        if not hasattr(self.algorithm, "get_next_action"):
            raise ValueError("Algorithm reuires a get_next_action function defined")

        return InteractiveDataLoader(
            self.dataset,
            action_function=self.algorithm.get_next_action
        )

    @classmethod
    def load_dataset(cls, config, train=True):
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        dataset_args = dict(config.get("dataset_args", {}))
        return dataset_class(**dataset_args)

    def run_iteration(self):
        return self.run_epoch()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "OffPolicyReinforcementLearning"
        # Extended methods
        eo["setup_experiment"].append(exp + ".setup_experiment")
        eo["get_state"] = exp + ": epoch, total steps, algorithm"
        eo["set_state"] = exp + ": epoch, total steps, algorithm"

        eo.update(
            # Overwritten methods
            run_iteration=[exp + ".run_iteration"],
            should_stop=[exp + ".should_stop"],
            # New methods
            collect_data=[exp + ".collect_data"],
            post_step=[exp + ".post_step"],
            create_loaders=[exp + ".create_loaders"],
            train_epoch=[exp + ".train_epoch"],
            run_epoch=[exp + ".run_epoch"],
            load_dataset=[exp + ".load_dataset"],
            pre_epoch=[exp + ".pre_epoch"],
            post_epoch=[exp + ".post_epoch"]
        )
