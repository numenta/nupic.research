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

from projects.imagenet.experiments_base import CONFIGS as C1
from projects.imagenet.experiments_composed_lr import CONFIGS as C5
from projects.imagenet.experiments_custom_super import CONFIGS as C4
from projects.imagenet.experiments_default import CONFIGS as C2
from projects.imagenet.experiments_superconvergence import CONFIGS as C3

"""
Import and collect all Imagenet experiment configurations into one CONFIG
"""

# Collect all configurations
CONFIGS = dict()
CONFIGS.update(C1)
CONFIGS.update(C2)
CONFIGS.update(C3)
CONFIGS.update(C4)
CONFIGS.update(C5)
