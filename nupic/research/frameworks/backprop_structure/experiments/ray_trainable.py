# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

from ray import tune


def as_ray_trainable(experiment_class):
    class Cls(tune.Trainable):
        def _setup(self, config):
            self.exp = experiment_class(**config)

        def _train(self):
            return self.exp.run_epoch(self.iteration)

        def _save(self, checkpoint_dir):
            return self.exp.save(checkpoint_dir)

        def _restore(self, checkpoint):
            self.exp.restore(checkpoint)

    Cls.__name__ = experiment_class.__name__

    return Cls
