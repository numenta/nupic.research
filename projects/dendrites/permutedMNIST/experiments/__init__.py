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

from .base import CONFIGS as BASE
from .centroid import CONFIGS as CENTROID
from .batch_mnist import CONFIGS as BATCH_MNIST
from .sp_context_search import CONFIGS as SP_PROTO
from .sp_context import CONFIGS as SP_CONTEXT
from .pmnist_no_dendrites import CONFIGS as PMNIST_NO_DENDRITES
from .batch_pmnist import CONFIGS as BATCH_PMNIST

"""
Import and collect all experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(CENTROID)
CONFIGS.update(BATCH_MNIST)
CONFIGS.update(SP_PROTO)
CONFIGS.update(SP_CONTEXT)
CONFIGS.update(PMNIST_NO_DENDRITES)
CONFIGS.update(BATCH_PMNIST)
