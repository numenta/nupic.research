# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from .default_base import CONFIGS as DEFAULT_BASE
from .sigopt_experiments import CONFIGS as SIGOPT_EXPERIMENTS
from .small_sparse import CONFIGS as SMALL_SPARSE
from .sparse_resnets import CONFIGS as SPARSE_RESNETS

CONFIGS = dict()
CONFIGS.update(DEFAULT_BASE)
CONFIGS.update(SPARSE_RESNETS)
CONFIGS.update(SIGOPT_EXPERIMENTS)
CONFIGS.update(SMALL_SPARSE)
