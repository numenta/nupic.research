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

import torch


class TestNoise(object):
    def __init__(self, noise_levels, noise_test_freq=0, noise_test_at_end=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.noise_test_freq = noise_test_freq
        self.noise_test_at_end = noise_test_at_end
        self.noise_levels = noise_levels

    def run_epoch(self, iteration):
        result = super().run_epoch(iteration)

        if ((self.noise_test_at_end
             and result["done"])
            or (self.noise_test_freq != 0
                and (iteration + 1) % self.noise_test_freq == 0)):

            noise_score = 0
            noise_results = {}
            for noise_level in self.noise_levels:
                loader = torch.utils.data.DataLoader(
                    self.dataset_manager.get_test_dataset(noise_level),
                    batch_size=self.batch_size_test,
                    shuffle=False,
                    pin_memory=torch.cuda.is_available()
                )
                noise_result = self.test(loader)
                noise_score += noise_result["total_correct"]
                noise_results[str(noise_level)] = noise_result

            result["noise_results"] = noise_results
            result["noise_score"] = noise_score

        return result
