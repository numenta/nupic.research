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
import numpy as np

from nupic.research.frameworks.pytorch.functions import k_winners2d_local
from nupic.torch.modules import KWinners2d


class KWinners2dLocal(KWinners2d):
    """
    A K-winner 2d module where the k-winners are chosen locally across all the
    channels
    """

    def __init__(self, channels, percent_on=0.1, k_inference_factor=1.5,
                 boost_strength=1.0, boost_strength_factor=0.9,
                 duty_cycle_period=1000):
        super().__init__(channels, percent_on, k_inference_factor, boost_strength,
                         boost_strength_factor, duty_cycle_period)
        self.k = int(round(self.channels * self.percent_on))
        self.k_inference = int(round(self.channels * self.percent_on_inference))

    def forward(self, x):
        if self.n == 0:
            self.n = np.prod(x.shape[1:])

        if self.training:
            x = k_winners2d_local(x, self.duty_cycle, self.k, self.boost_strength)
            self.update_duty_cycle(x)
        else:
            x = k_winners2d_local(x, self.duty_cycle, self.k_inference,
                                  self.boost_strength)
        return x
