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

__all__ = [
    "Profile",
]


class Profile:
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.epoch_pr = cProfile.Profile()
        self.epoch_pr.enable()

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        self.epoch_pr.disable()
        filepath = os.path.expanduser(f"~/profile-epoch{self.current_epoch}.profile")
        pstats.Stats(self.epoch_pr).dump_stats(filepath)
        print(f"Saved {filepath}")
        del self.epoch_pr
