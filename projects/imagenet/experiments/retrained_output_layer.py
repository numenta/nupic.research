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

from copy import deepcopy

import numpy as np

from nupic.research.frameworks.pytorch.restore_utils import load_multi_state

from .custom_super import CONFIGS


def keep_1st_and_last_checkpoints(iteration):
    """
    Score the first and last iterations the highest.
    """
    if iteration == 0:
        return np.inf
    else:
        return iteration


REFERENCE_SUPER_SPARSE_100 = deepcopy(CONFIGS["super_sparse_100"])
REFERENCE_SUPER_SPARSE_100.update(dict(
    checkpoint_at_start=True,
    checkpoint_at_end=True,
    keep_checkpoints_num=2,
    checkpoint_freq=3,
    checkpoint_score_attr="checkpoint_score",
    checkpoint_scoring_function=keep_1st_and_last_checkpoints,
))


NEW_OUTPUT_INIT_SUPER_SPARSE_100 = deepcopy(REFERENCE_SUPER_SPARSE_100)
NEW_OUTPUT_INIT_SUPER_SPARSE_100.update(dict(
    modify_init_hook=load_multi_state,
    restore_nonlinear="chkpt_40",
))


REUSE_OUTPUT_INIT_SUPER_SPARSE_100 = deepcopy(REFERENCE_SUPER_SPARSE_100)
REUSE_OUTPUT_INIT_SUPER_SPARSE_100.update(dict(
    modify_init_hook=load_multi_state,
    restore_linear="chkpt_0",
    restore_nonlinear="chkpt_40",
))


# Export all configurations
CONFIGS = dict(
    reference_super_sparse_100=REFERENCE_SUPER_SPARSE_100,
    new_output_init_super_sparse_100=NEW_OUTPUT_INIT_SUPER_SPARSE_100,
    resuse_output_init_super_sparse_100=REUSE_OUTPUT_INIT_SUPER_SPARSE_100,
)
