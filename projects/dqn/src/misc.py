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


class Scheduler():
    """Generic scheduler class for any rate"""

    def __init__(self, start_rate, end_rate, max_steps, num_updates=None):
        self.rate = start_rate
        self.end_rate = end_rate

        if num_updates is None:
            self.step_size = (end_rate - start_rate) / max_steps
        else:
            steps_per_update = int(max_steps * num_updates)
            self.step_size = (end_rate - start_rate) / steps_per_update

    def step(self):
        self.rate -= self.step_size

    def state_dict(self):
        return {
            "rate": self.rate,
            "step_size": self.step_size
        }

    def load_state_dict(self, state_dict):
        self.rate = state_dict["rate"]
        self.step_size = state_dict["step_size"]

    def __call__(self):
        return self.rate
