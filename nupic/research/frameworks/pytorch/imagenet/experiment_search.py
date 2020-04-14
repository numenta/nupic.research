#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import os
import pickle
from copy import deepcopy

import numpy as np


class SearchOption:
    """
    Allow usage of different search options for exploration of hyperparameters space
    In-place substitute for Ray Tune heuristic search options
    """
    def __init__(self, elements):
        self.elements = elements


class SequentialSearch(SearchOption):
    """Pick sequentially from a list with size equal to number of trials"""

    def expand_in_place(self, config, index, k1, k2=None):
        """
        Updates config in place with one of the elements in a list

        :param config: (dict) Config (or kwargs) from the experiment
        :param index: (int) Position of the element to be selected in the list
        :param k1: (str) Argument from config that will be updated
        :param k2: (str) Argument from config that will be updated. Secondary key
                         used if argument to be updated is in an nested dictionary.
                         k1 is the key for the outer dict, and k2 for the inner dict
        """

        if index >= len(self.elements):
            raise ValueError(
                "Number of elements in SequentialSearch shorter "
                "than number of trials."
            )
        element = self.elements[index]
        print(f"***** seed {element}")

        # assign
        if k2 is None:
            config[k1] = element
        else:
            config[k1][k2] = element


class RandomSearch(SearchOption):
    """Sample from a list of options or a stochastic function"""

    def expand_in_place(self, config, k1, k2=None):
        """
        Updates config in place with a random element selected from a list or function.

        :param config: (dict) Config (or kwargs) from the experiment
        :param k1: (str) Argument from config that will be updated
        :param k2: (str) Argument from config that will be updated. Secondary key
                         used if argument to be updated is in an nested dictionary.
                         k1 is the key for the outer dict, and k2 for the inner dict
        """
        if callable(self.elements):
            element = self.elements()
        elif hasattr(self.elements, "__iter__"):
            element = np.random.choice(self.elements)
        else:
            raise ValueError(
                "RandomSearch requires a function with no args or an iterator"
            )

        # assign
        if k2 is None:
            config[k1] = element
        else:
            config[k1][k2] = element


class GridSearch(SearchOption):
    """Expand one experiment per option"""

    def expand_to_list(self, config, k1, k2=None):
        """
        Returns a list of experiment, with each one containing one of the
        possible elements of the grid search.

        :param config: (dict) Config (or kwargs) from the experiment
        :param k1: (str) Argument from config that will be updated
        :param k2: (str) Argument from config that will be updated. Secondary key
                         used if argument to be updated is in an nested dictionary.
                         k1 is the key for the outer dict, and k2 for the inner dict
        :return: returns a copy of the original config with the parameter updated
        """
        expanded_list = []
        for el in self.elements:
            expanded_config = deepcopy(config)
            if k2 is None:
                expanded_config[k1] = el
            else:
                expanded_config[k1][k2] = el
            expanded_list.append(expanded_config)

        return expanded_list


class TrialsCollection:
    """
    Create imagenet experiment model with option to load state from checkpoint
    Expand config with different search options (e.g grid search, random search)
    into a list of trials

    :param config: (dict) Config (or kwargs) from the original experiment
    :param num_trials: (int) Number of trials required for each configuration.
                             Equivalent to num_samples on Ray
    :param restore: (bool) Whether or not continue from previous experiment,
                           based on experiment_name
    """

    def __init__(self, config, num_trials, restore=True):
        # all experiments are required a name for later retrieval
        if "experiment_name" not in config:
            self.name = "".join([chr(np.random.randint(97, 123)) for _ in range(10)])
        else:
            self.name = config["experiment_name"]
        print(f"***** Experiment {self.name} started")

        self.base_config = config
        self.num_trials = num_trials
        self.path = config["local_dir"]
        self.path_pending = os.path.join(self.path, self.name + "_pending.p")
        self.path_completed = os.path.join(self.path, self.name + "_completed.p")

        if restore and os.path.exists(self.path_pending):
            self.restore()
        else:
            self.pending = self.expand_trials(config, num_trials)
            self.completed = []
            self.save()

        self.total_trials = len(self.pending) + len(self.completed)

    def report_progress(self):
        """Report number of experiments completed"""
        print(f"***** Trials completed: {len(self.completed)}/{self.total_trials}")

    def retrieve(self):
        """Remove a trial from pending list and return"""
        while len(self.pending) > 0:
            trial = self.pending.pop()
            yield trial

    def mark_completed(self, trial, save=True):
        """Add a trial to completed list and save"""
        self.completed.append(trial)
        if save:
            self.save()

    def save(self):
        """Save pending and completed trials in results folder"""
        with open(self.path_pending, "wb") as f:
            pickle.dump(self.pending, f)
        with open(self.path_completed, "wb") as f:
            pickle.dump(self.completed, f)

    def restore(self):
        """Restore pending and completed trials from results folder"""
        with open(self.path_pending, "rb") as f:
            self.pending = pickle.load(f)
        with open(self.path_completed, "rb") as f:
            self.completed = pickle.load(f)

    @staticmethod
    def expand_trials(base_config, num_samples=1): # noqa: C901
        """
        Convert experiments using SearchOption into a list of experiments
        List of experiments can be executed in parallel or sequentially

        Iterates through main dictionary, and through dictionaries within the main dict
        This picks up nested arguments such as in optimizer_args and lr_scheduler
        Only supports two levels
        TODO: replace for cleaner and more flexible recursive approach
        """

        # expand all grid search
        stack = [base_config]
        trials = []
        while len(stack) != 0:
            config = stack.pop()
            pending_analysis = False
            # first level
            for k1, v1 in config.items():
                if pending_analysis:
                    break
                elif v1.__class__ == GridSearch:
                    stack.extend(v1.expand_to_list(config, k1))
                    pending_analysis = True
                elif type(v1) == dict:
                    # second level
                    for k2, v2 in v1.items():
                        if pending_analysis:
                            break
                        elif v2.__class__ == GridSearch:
                            stack.extend(v2.expand_to_list(config, k1, k2))
                            pending_analysis = True

            if not pending_analysis:
                trials.append(config)

        # multiply by num_samples
        expanded_trials = []
        for trial in trials:
            for _ in range(num_samples):
                expanded_trials.append(deepcopy(trial))

        # replace all sample from and random search
        for index, config in enumerate(expanded_trials):
            # first level
            for k1, v1 in config.items():
                if v1.__class__ == RandomSearch:
                    v1.expand_in_place(config, k1)
                elif v1.__class__ == SequentialSearch:
                    v1.expand_in_place(config, index, k1)
                elif type(v1) == dict:
                    # second level
                    for k2, v2 in v1.items():
                        if v2.__class__ == RandomSearch:
                            v2.expand_in_place(config, k1, k2)
                        elif v2.__class__ == SequentialSearch:
                            v2.expand_in_place(config, index, k1, k2)

        return expanded_trials
