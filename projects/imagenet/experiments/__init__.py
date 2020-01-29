#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

from .base import CONFIGS as BASE
from .composed_lr import CONFIGS as COMPOSED_LR
from .custom_super import CONFIGS as CUSTOM_SUPER
from .default import CONFIGS as DEFAULT
from .mixed_precision import CONFIGS as MIXED_PRECISION
from .super_convergence import CONFIGS as SUPER_CONVERGENCE
from .custom_25 import CONFIGS as CUSTOM_25
from .sparse_r1 import CONFIGS as SPARSE_R1

"""
Import and collect all Imagenet experiment configurations into one CONFIG
"""
__all__ = ["CONFIGS"]

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(BASE)
CONFIGS.update(COMPOSED_LR)
CONFIGS.update(CUSTOM_SUPER)
CONFIGS.update(DEFAULT)
CONFIGS.update(MIXED_PRECISION)
CONFIGS.update(SUPER_CONVERGENCE)
CONFIGS.update(CUSTOM_25)
CONFIGS.update(SPARSE_R1)
