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

import cProfile
import os
import pstats


class Profile:
    """
    Save cProfile traces for initialization and each run_epoch.
    """
    def __init__(self):
        super().__init__()
        self.execution_order["setup_experiment"].insert(0, "Profile begin")
        self.execution_order["setup_experiment"].append("Profile end")
        self.execution_order["run_epoch"].insert(0, "Profile begin")
        self.execution_order["run_epoch"].append("Profile end")

    def setup_experiment(self, config):
        self.use_cProfile = (self.rank == 0)
        if self.use_cProfile:
            pr = cProfile.Profile()
            pr.enable()

        super().setup_experiment(config)

        if self.use_cProfile:
            pr.disable()
            filepath = os.path.join(self.logdir,
                                    f"profile-initialization.profile")
            pstats.Stats(pr).dump_stats(filepath)
            self.logger.info(f"Saved {filepath}")

    def run_epoch(self):
        if self.use_cProfile:
            pr = cProfile.Profile()
            pr.enable()

        result = super().run_epoch()

        if self.use_cProfile:
            pr.disable()
            filepath = os.path.join(self.logdir,
                                    f"profile-epoch{self.epoch}.profile")
            pstats.Stats(pr).dump_stats(filepath)
            self.logger.info(f"Saved {filepath}")

        return result
