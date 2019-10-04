#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import unittest

from nupic.research.frameworks.dynamic_sparse.models import BaseModel
from nupic.research.frameworks.dynamic_sparse.networks import MLPHeb


class BaseModelTest(unittest.TestCase):
    def test_post_epoch_updates(self):
        """Ensure boost strength is updated in post_epoch."""
        model = BaseModel(
            network=MLPHeb(
                config=dict(
                    percent_on_k_winner=[0.2, 1.0, 0.1],
                    boost_strength=[1.4, 1.5, 1.6],
                    boost_strength_factor=[0.7, 0.8, 0.9],
                )
            ),
            config=dict(on_perc=0.1, hebbian_prune_perc=0.25, weight_prune_perc=0.50),
        )
        model.setup()

        self.assertEqual(model.network.classifier[1][1].percent_on, 0.2)
        self.assertEqual(model.network.classifier[3][1].percent_on, 0.1)
        self.assertEqual(model.network.classifier[1][1].boost_strength, 1.4)
        self.assertEqual(model.network.classifier[3][1].boost_strength, 1.6)
        self.assertEqual(model.network.classifier[1][1].boost_strength_factor, 0.7)
        self.assertEqual(model.network.classifier[3][1].boost_strength_factor, 0.9)

        model._post_epoch_updates()

        self.assertAlmostEqual(
            float(model.network.classifier[1][1].boost_strength), 1.4 * 0.7, places=5
        )
        self.assertAlmostEqual(
            float(model.network.classifier[3][1].boost_strength), 1.6 * 0.9, places=5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
