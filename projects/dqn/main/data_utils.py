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


from collections import deque, namedtuple

from torch.utils.data import Dataset
from torchvision import transforms

Experience = namedtuple("Experience", (
    "observation", "action", "reward", "next_observation", "done", "info")
)


class GymEnvDataset(Dataset):
    """
    Based on gym
    TODO: implement env dataset wth stacking frames capability
          number of channels in observation space might change if frames are stacked
          can be done with transforms and/or a small deque buffer.
    """

    def __init__(
        self,
        env_init_func,
        env_args,
        transform=None,
        num_frames=None
    ):
        self.env = env_init_func(**env_args)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.unsqueeze(dim=0)),
            ])
        else:
            self.transform = transform

        self.episode_complete = False

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        return self.transform(self.env.reset())

    def terminate_env(self):
        self.env.close()

    def __getitem__(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_complete = done

        if self.transform:
            observation = self.transform(observation)

        return observation, reward, done, info

    def __len__(self):
        return self.env._max_episode_steps


class InteractiveDataLoader():

    def __init__(self, dataset, action_function):
        self.dataset = dataset
        self.action_function = action_function

    def reset(self):
        """Resets environment associated with dataloader"""
        self.last_observation = self.dataset.reset()

    def __iter__(self):
        """
        Note: first step should return all Nones except for observation, given it is the
        environment initialization
        """
        self.reset()
        while not self.dataset.episode_complete:
            action = self.action_function(self.last_observation)
            next_observation, reward, done, info = self.dataset[action]
            experience = Experience(
                self.last_observation,
                action,
                reward,
                done,
                next_observation,
                info
            )
            self.last_observation = next_observation
            yield experience


class ReplayBuffer(Dataset):

    def __init__(self, buffer_size=1000000, transform=None):
        self.experiences = deque(maxlen=buffer_size)
        self.transform = transform

    def __len__(self):
        return len(self.experiences)

    def extend(self, experiences):
        self.experiences.extend(experiences)

    def append(self, experience):
        self.experiences.append(experience)

    def __getitem__(self, index):
        sample = self.experiences[index]
        if self.transform:
            sample = self.transform(sample)

        return sample
