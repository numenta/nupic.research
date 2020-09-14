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

from nupic.research.frameworks.sigopt import mixins

from .sigopt_experiment import SigOptExperiment


class SigOptSGDOneCycleLRExperiment(SigOptExperiment,
                                    mixins.SGDParams,
                                    mixins.OneCycleLRParams):
    """Tune hyper-parameters using SGD and OneCycleLR."""
    pass


class SigOptSGDStepLRExperiment(SigOptExperiment,
                                mixins.SGDParams,
                                mixins.StepLRParams):
    """Tune hyper-parameters using SGD and StepLR."""
    pass


__all__ = [
    "SigOptSGDOneCycleLRExperiment",
    "SigOptSGDStepLRExperiment",
]
