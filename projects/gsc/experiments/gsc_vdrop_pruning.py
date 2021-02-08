#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Run a simple GSC experiment using variational dropout modules.
"""

from copy import deepcopy

import torch

from .base import DEFAULT_BASE
from nupic.research.frameworks.backprop_structure.networks import gsc_lenet_vdrop
from nupic.research.frameworks.vernon.distributed import experiments, mixins

class GSCVDropExperiment(mixins.RegularizeLoss,
                           mixins.ConstrainParameters,
                           experiments.SupervisedExperiment):
    pass


GSC_VDROP = deepcopy(DEFAULT_BASE)
GSC_VDROP.update(
    experiment_class = GSCVDropExperiment,
    # Training batch size
    batch_size=32,
    # Validation batch size
    val_batch_size=32,
    gamma_postwarmup= 0.9553489086781097,
    gamma_prewarmup= 1.0,
    gamma_warmup= 0.8086114832044939,
    log2_batch_size= 5,
    log_reg_factor_end= -2.2155542642611037,
    log_reg_factor_start= -3.8539378170655567,
    lr= 0.0011321979071206214,
    num_epochs= 167,
    reg_warmup_end_epoch= 111,
    reg_warmup_start_epoch= 108,
    model_class = gsc_lenet_vdrop,
)

CONFIGS = dict(
    gsc_vdrop=GSC_VDROP,
)