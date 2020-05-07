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
    def setup_experiment(self, config):
        pr = cProfile.Profile()
        pr.enable()
        super().setup_experiment(config)
        pr.disable()
        pstats.Stats(pr).dump_stats(
            os.path.join(self.logdir, f"profile-initialization.profile")
        )

    def run_epoch(self, epoch):
        pr = cProfile.Profile()
        pr.enable()
        result = super().run_epoch(epoch)
        pr.disable()
        pstats.Stats(pr).dump_stats(
            os.path.join(self.logdir, f"profile-epoch{epoch}.profile")
        )

        return result
