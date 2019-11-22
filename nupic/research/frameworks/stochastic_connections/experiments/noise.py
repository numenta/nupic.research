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

from .vanilla import Vanilla


class Noise(Vanilla):
    def __init__(self, noise_test_epochs, **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.noise_test_epochs = noise_test_epochs
        self.noise_levels = kwargs["dataset_config"]["noise_level"]

    def run_epoch(self, iteration):
        result = super(Noise, self).run_epoch(iteration)

        noise_score = 0

        noise_results = {}
        if iteration in self.noise_test_epochs:
            for noise_level in self.noise_levels:
                noise_result = self.test(self.dataset.noise_loader)
                noise_score += noise_result["total_correct"]
                noise_results[str(noise_level)] = noise_result
                print("Noise level {}, result {}".format(noise_level, noise_result))

        result["noise_results"] = noise_results
        result["noise_score"] = noise_score
        return result


class NoiseRay(tune.Trainable):
    def _setup(self, config):
        self.exp = Noise(**config)

    def _train(self):
        return self.exp.run_epoch(self.iteration)

    def _save(self, checkpoint_dir):
        return self.exp.save(checkpoint_dir)

    def _restore(self, checkpoint):
        self.exp.restore(checkpoint)
