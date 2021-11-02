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
import warnings

import nupic.torch.modules


class KWinners2dLocal(nupic.torch.modules.KWinners2d):
    """
    A K-winner 2d module where the k-winners are chosen locally across all the
    channels
    """

    def __init__(self, channels, percent_on=0.1, k_inference_factor=1.5,
                 boost_strength=1.0, boost_strength_factor=0.9,
                 duty_cycle_period=1000):
        super().__init__(channels=channels, percent_on=percent_on,
                         k_inference_factor=k_inference_factor,
                         boost_strength=boost_strength,
                         boost_strength_factor=boost_strength_factor,
                         duty_cycle_period=duty_cycle_period, local=True)
        warnings.warn("KWinners2dLocal moved to nupic.torch. This class will "
                      "soon be removed from nupic.research", DeprecationWarning)
