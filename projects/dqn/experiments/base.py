#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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

"""
Base RL Experiment configuration.
"""

import os

import gym

from nupic.research.frameworks.ray.trainables import DebugTrainable
from main import DQN, GymEnvDataset, OffPolicyReinforcementLearningExperiment

def gym_env(**env_args):
    return gym.make(**env_args)


base = dict()


dqn_base = dict(
    logdir=os.path.expanduser("~/nta/results"),
    ray_trainable=DebugTrainable,
    algorithm_class=DQN,
    experiment_class=OffPolicyReinforcementLearningExperiment,
    dataset_class=GymEnvDataset,
    dataset_args=dict(
        env_init_func=gym_env,
        env_args=dict(
            id="Breakout-v4"
        ),
    ),
    reward_clip=(-1.0, +1.0),
    num_steps=int(1e6),
    buffer_batch_size=128,
)


# Export configurations in this file
CONFIGS = dict(
    base=base,
    dqn_base=dqn_base,
)
