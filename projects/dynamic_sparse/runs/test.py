# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

# import sys
# sys.path.append("..")

# from models.base_models import BaseModel
# from .. import models

# from dynamic_sparse.models import *


import os
import sys

sys.path.append("../../")
# sys.path.append(os.path.expanduser("~/nta/nupic.research/projects/"))

# import dynamic_sparse.models as models
# import dynamic_sparse.networks as networks
# from dynamic_sparse.common import *

print("test")
print(__file__.replace(".py", ""))
print("Current File Name : ", os.path.basename(__file__))
print("Current File Path : ", os.path.realpath(__file__))

# PYTHONPATH=~/nta/nupic.research/projects/ python mlp_heb.py
