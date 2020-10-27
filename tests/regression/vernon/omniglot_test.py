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

import unittest

from torchvision.datasets import Omniglot

ROOT = "/mnt/datasets"


class OmniglotTest(unittest.TestCase):

    def test_load_dataset(self):
        """Load the background and evaluation set without downloading them."""
        background_dataset = Omniglot(
            root=ROOT,
            background=True,
            download=False,
            transform=None
        )
        self.assertEqual(len(background_dataset), 19280)

        evaluation_dataset = Omniglot(
            root=ROOT,
            background=False,
            download=False,
            transform=None
        )
        self.assertEqual(len(evaluation_dataset), 13180)


if __name__ == "__main__":
    unittest.main(verbosity=2)
