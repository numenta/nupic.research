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

from collections import deque

import ray
import torch


class AxService:
    """
    This will often be a subclass of AxClient or CoreAxClient that simply
    overrides its __init__ method. The __init__ method takes one parameter: the
    serialization file path. The user can choose whether or not to support
    serialization.
    """
    def __init__(self, ax_client_class, serialized_filepath=None,  # NOQA: C901
                 actor_resources=None):
        """
        @param actor_resources (dict or None)
        Custom resources for the backend and frontend. This is useful for
        ensuring that the AxClient runs on a reliable machine (e.g. the head EC2
        on-demand instance rather than a worker spot instance).
        """
        @ray.remote(num_gpus=(1 if torch.cuda.is_available() else 0),
                    resources=actor_resources)
        class AxBackend(ax_client_class):
            def __init__(self):
                super().__init__(serialized_filepath)
                self.frontend = None

            def add_frontend(self, frontend):
                self.frontend = frontend

            def queue_incomplete_trials(self):
                for trial in self.experiment.trials.values():
                    if not trial.status.is_completed:
                        self.frontend.put_params.remote((trial.arm.parameters,
                                                         trial.index))

            def get_num_completed_trials(self):
                return sum((1 if trial.status.is_completed else 0)
                           for trial in self.experiment.trials.values())

            def _absorb_results_queue(self):
                results = ray.get(self.frontend.get_and_clear_results.remote())
                for trial_index, raw_data, metadata in results:
                    self.complete_trial(trial_index, raw_data, metadata)
                self.save()

            def notify_results_available(self):
                self._absorb_results_queue()

            def notify_params_needed(self):
                # In the time since the head invoked this method, they may have invoked
                # notify_results_available, but that actor invocation is queued after
                # this one. It is best to have all of the available results when
                # generating the next parameters, so check if there are any available.
                self._absorb_results_queue()

                params, trial_index = self.get_next_trial()
                self.frontend.put_params.remote((params, trial_index))
                self.save()

        AxBackend.__name__ = f"{ax_client_class.__name__}Backend"

        @ray.remote(num_cpus=0, resources=actor_resources)
        class AxFrontend:
            """
            Serves as a fast-responding head.
            """
            def __init__(self, backend):
                self.backend = backend
                self.params_queue = deque()
                self.results_queue = deque()
                self.notified_params_needed = False
                self.notified_results_available = False

            def get_next_trial(self):
                if len(self.params_queue) > 0:
                    return True, self.params_queue.pop()
                else:
                    if not self.notified_params_needed:
                        self.backend.notify_params_needed.remote()
                        self.notified_params_needed = True
                    return False, None

            def complete_trial(self, trial_index, raw_data, metadata=None):
                self.results_queue.append((trial_index, raw_data, metadata))
                if not self.notified_results_available:
                    self.backend.notify_results_available.remote()
                    self.notified_results_available = True

            def put_params(self, params):
                self.params_queue.appendleft(params)
                self.notified_params_needed = False

            def get_and_clear_results(self):
                results = list(self.results_queue)
                self.results_queue.clear()
                self.notified_results_available = False
                return results

        self.backend = AxBackend.remote()
        self.frontend = AxFrontend.remote(self.backend)
        self.backend.add_frontend.remote(self.frontend)
