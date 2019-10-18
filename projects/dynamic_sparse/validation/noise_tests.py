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

# script to run noise tests 

# load the model and the configurations
# could then run a specific model 

import os
import pickle
from nupic.research.frameworks.dynamic_sparse.models import SparseModel
from nupic.research.frameworks.dynamic_sparse.common.datasets import Dataset

# load
model = SparseModel(config=dict(load_from_checkpoint=True, device='cpu'))
dataset = Dataset(model.config)
noise_levels = [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]

# calculate
results = []
for noise_level in noise_levels:
    dataset.set_noise_loader(noise_level)
    loss, acc = model.evaluate(dataset)
    results.append((noise_level, loss, acc))

# save
save_path = os.path.join(os.path.expanduser("~/nta/results"), "noise_tests")
if not os.path.exists(save_path):
    os.mkdir(save_path)
pickle.dump(results, os.path.join(save_path, 'results.txt'))

