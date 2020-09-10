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

import mixins

from .sigopt_experiment import SigOptExperiment


class SigOptImagenetExperiment(SigOptExperiment, mixins.ImagenetParams):
    """
    A subclass of SigOptExperiment used to sit between an experiment runner (such as
    Ray) and the ImagenetExperiment class. update_config_with_suggestion() is specific
    to our ImagenetExperiment config.
    """
    pass


class SigOptSDGOneCycleLRExperiment(SigOptExperiment,
                                    mixins.SGDParams,
                                    mixins.OneCycleLRParams):
    """Tune hyper-parameters using SDG and OneCycleLR."""
    pass


class SigOptSDGStepLRExperiment(SigOptExperiment, mixins.SGDParams, mixins.StepLRParams):
    """Tune hyper-parameters using SDG and StepLR."""
    pass
