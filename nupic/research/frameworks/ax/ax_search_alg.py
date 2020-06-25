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

import logging

import ray
from ray.tune.suggest.suggestion import SuggestionAlgorithm

logger = logging.getLogger(__name__)


class NonblockingAxSearch(SuggestionAlgorithm):
    def __init__(self, ax_frontend, metric_function=None, max_concurrent=40,
                 m_suggestions_allowed_before_nth_completion=None, **kwargs):
        """
        @param metric_function (function or None)
        An optional function that takes the trial's result and returns a result
        for Ax.

        @param m_suggestions_allowed_before_nth_completion (None or pair of ints)
        Useful for making sure you don't try to generate a non-random set of
        parameters before any trials have completed. (Ax will throw an error if
        you do this.) Ax also gives you an alternate way of dealing with this:
        if you set the min_trials_observed in the SOBOL GenerationStep and set
        its enforce_num_trials to False, Ax will instead continue generating
        random trials (beyond your requested number of random trials) until it
        receives results for min_trials_observed.
        """
        assert type(max_concurrent) is int and max_concurrent > 0
        self._ax_frontend = ax_frontend
        self._live_index_mapping = {}
        self._metric_function = metric_function
        self._max_concurrent = max_concurrent
        self._num_completed = 0
        self._num_suggested = 0
        self._m_suggestions_allowed_before_nth_completion = (
            m_suggestions_allowed_before_nth_completion
        )

        super().__init__(**kwargs)

    def suggest(self, trial_id):
        if self._num_live_trials() >= self._max_concurrent:
            return None

        if self._m_suggestions_allowed_before_nth_completion is not None:
            m, n = self._m_suggestions_allowed_before_nth_completion
            if self._num_completed < n and self._num_suggested >= m:
                return None

        success, v = ray.get(self._ax_frontend.get_next_trial.remote())
        if success:
            parameters, trial_index = v
            self._live_index_mapping[trial_id] = trial_index
            self._num_suggested += 1
            return parameters
        else:
            return None

    def on_trial_complete(self,
                          trial_id,
                          result=None,
                          error=False,
                          early_terminated=False):
        if not error and not early_terminated:
            trial_index = self._live_index_mapping[trial_id]
            if self._metric_function is not None:
                result = self._metric_function(result)

            self._ax_frontend.complete_trial.remote(
                trial_index, result,
                # Give future humans browsing the Ax results an easy way to
                # cross-reference back to the ray results.
                {"ray_trial_id": trial_id}
            )
            self._num_completed += 1
        self._live_index_mapping.pop(trial_id)

    def _num_live_trials(self):
        return len(self._live_index_mapping)
