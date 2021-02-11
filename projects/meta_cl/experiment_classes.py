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

from mixins import UpdateDendriteBoostStrength
from nupic.research.frameworks.vernon import MetaContinualLearningExperiment, mixins


class OMLExperiment(mixins.OnlineMetaLearning,
                    MetaContinualLearningExperiment):
    pass


class DendritesExperiment(mixins.RezeroWeights,
                          mixins.OnlineMetaLearning,
                          MetaContinualLearningExperiment):
    """
    This is similar to OMLExperiment, but now there is rezero weights.
    """
    pass


class BoostedDendritesExperiment(UpdateDendriteBoostStrength,
                                 mixins.RezeroWeights,
                                 mixins.OnlineMetaLearning,
                                 MetaContinualLearningExperiment):
    """
    This is similar to DendritesExperiment, but with an added mixin to update the boost
    strength of a DendriteSegementsBooster module.
    """
    pass


class IIDTrainingOMLExperiment(OMLExperiment):
    def sample_slow_data(self, tasks):
        """
        Return no data so that all of the images and targets
        can be sampled i.i.d. from the replay set.
        """
        return [], []

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "IIDTrainingOMLExperiment: "
        eo["sample_slow_data"] = [exp + "only use replay data in outer loop."]
        return eo
