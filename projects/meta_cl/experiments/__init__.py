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

from .metacl import CONFIGS as METACL
from .metacl_sigopt import CONFIGS as METACL_SIGOPT
from .metacl_track_stats import CONFIGS as METACL_TRACK_STATS
from .anml_replicate import CONFIGS as ANML_REPLICATE
from .anml_variants import CONFIGS as ANML_VARIANTS
from .dendrites import CONFIGS as DENDRITES
from .dendrite_variants import CONFIGS as DENDRITE_VARIANTS
from .oml_replicate import CONFIGS as OML_REPLICATE
from .oml_variants import CONFIGS as OML_VARIANTS
from .oml_regression_test import CONFIGS as OML_REGRESSION_TEST

"""
Import and collect all Imagenet experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(METACL)
CONFIGS.update(METACL_TRACK_STATS)
CONFIGS.update(METACL_SIGOPT)
CONFIGS.update(ANML_REPLICATE)
CONFIGS.update(ANML_VARIANTS)
CONFIGS.update(DENDRITES)
CONFIGS.update(DENDRITE_VARIANTS)
CONFIGS.update(OML_REPLICATE)
CONFIGS.update(OML_VARIANTS)
CONFIGS.update(OML_REGRESSION_TEST)
